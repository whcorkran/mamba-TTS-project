"""
Audio preprocessing for TTS model data loading.
Based on ControlSpeech/baseline/promptTTS/utils/audio.
"""
# some of the dependencies have deprecated apis and are not maintained, suppress warnings to not interrupt training
import warnings

warnings.filterwarnings("ignore")
from typing import Optional, Tuple, Union

import os
from pathlib import Path
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile
import torch
import torchaudio

from lib.naturalspeech3_facodec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download


# Optional, FACodec has built in preprocessing logic 
class BaseAudioPreprocessor:
    """Audio preprocessing: loading, resampling, normalization, silence trimming."""
 
    def __init__(
        self,
        sample_rate: int = 16000,
        loudness_norm: bool = True,
        target_loudness: float = -20.0,
        silence_trim: bool = True,
        trim_top_db: int = 20,
        peak_norm: bool = True,
    ):
        self.sample_rate = sample_rate
        self.loudness_norm = loudness_norm
        self.target_loudness = target_loudness
        self.silence_trim = silence_trim
        self.trim_top_db = trim_top_db
        self.peak_norm = peak_norm
    
    def load_audio(self, path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        sr = sr or self.sample_rate
        wav, _ = librosa.core.load(path, sr=sr)
        return wav, sr
    
    def resample(self, wav: np.ndarray, orig_sr: int, target_sr: Optional[int] = None) -> np.ndarray:
        """Resample audio to target sample rate."""
        target_sr = target_sr or self.sample_rate
        if orig_sr != target_sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
        return wav
    
    def normalize_loudness(self, wav: np.ndarray, sr: Optional[int] = None, target_db: Optional[float] = None) -> np.ndarray:
        """Normalize audio loudness using ITU-R BS.1770."""
        sr = sr or self.sample_rate
        target_db = target_db if target_db is not None else self.target_loudness
        
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        
        if np.isinf(loudness):  # Silent audio
            return wav
        
        wav = pyln.normalize.loudness(wav, loudness, target_db)
        if np.abs(wav).max() > 1.0:
            wav = wav / np.abs(wav).max()
        return wav
    
    def normalize_peak(self, wav: np.ndarray) -> np.ndarray:
        """Normalize audio to peak amplitude of 1.0."""
        max_val = np.abs(wav).max()
        return wav / max_val if max_val > 0 else wav
    
    def trim_silence(self, wav: np.ndarray, top_db: Optional[int] = None) -> np.ndarray:
        """Trim leading and trailing silence."""
        top_db = top_db if top_db is not None else self.trim_top_db
        wav, _ = librosa.effects.trim(wav, top_db=top_db)
        return wav
    
    def preprocess(self, path_or_wav: Union[str, np.ndarray], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Full preprocessing pipeline."""
        if isinstance(path_or_wav, str):
            wav, sr = self.load_audio(path_or_wav)
        else:
            wav = path_or_wav
            sr = sr or self.sample_rate
            wav = self.resample(wav, sr)
            sr = self.sample_rate
        
        if self.loudness_norm:
            wav = self.normalize_loudness(wav, sr)
        if self.silence_trim:
            wav = self.trim_silence(wav)
        if self.peak_norm:
            wav = self.normalize_peak(wav)
        
        return wav, sr
    
    def save_wav(self, wav: np.ndarray, path: str, sr: Optional[int] = None, normalize: bool = False) -> None:
        """Save audio to WAV file."""
        sr = sr or self.sample_rate
        if normalize:
            wav = self.normalize_peak(wav)
        wav_int16 = (wav * 32767).astype(np.int16)
        if not path.endswith('.wav'):
            path = path.rsplit('.', 1)[0] + '.wav'
        wavfile.write(path, sr, wav_int16)

class FACodecEncoder:
    """
    FACodec pretrained audio codec encoder (FROZEN - NOT TRAINED)
    Source: https://github.com/lifeiteng/naturalspeech3_facodec.git
    
    This is a PRETRAINED model downloaded from HuggingFace that is used frozen:
    - Encoder: Converts audio → continuous representations
    - Decoder: Quantizes into discrete codec tokens (6 quantizers total)
    - Output: Codec tokens (B, T, 6) + Timbre embeddings (B, 256)
    
    Timbre Embeddings vs Style Vectors:
    - FACodec outputs 256-dim timbre embeddings (speaker identity)
    - ControlSpeech needs 512-dim style vectors (timbre + prosody + emotion)
    - Solution: Project 256→512 with learned linear layer
    - This projection is TRAINABLE (part of your model)
    
    Note: FACodec itself is frozen and reused from pretrained checkpoint.
    Only the 256→512 projection layer is trained with your TTS model.
    """

    def __init__(self, max_seq_len: int = 1024, device: str = None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use FACodec V2 classes with V2 checkpoints from HuggingFace
        # (from official test.py in naturalspeech3_facodec repo)
        self.fa_encoder = FACodecEncoderV2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.fa_decoder = FACodecDecoderV2(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        encoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin"
        )
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin"
        )

        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=True))
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt, weights_only=True))

        # Move to device and set eval mode
        self.fa_encoder.to(self.device).eval().requires_grad_(False)
        self.fa_decoder.to(self.device).eval().requires_grad_(False)
        
        # Projection layer to match ControlSpeech paper's 512-dim style vector
        # This is REQUIRED by the architecture (not optional)
        import torch.nn as nn
        self.style_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
        )
        # Initialize projection weights
        nn.init.xavier_uniform_(self.style_projection[0].weight)
        nn.init.zeros_(self.style_projection[0].bias)
        # Move projection to device
        self.style_projection.to(self.device)

    def encode(self, wav: str | bytes | list[str | bytes], sr: int = 16000):
        """Encode audio file(s) to FACodec tokens.
        
        Args:
            wav: Path(s) to audio file(s), or raw bytes from audio file(s)
            sr: Target sample rate
        """
        import io
        import soundfile as sf
        
        # Normalize to list
        if isinstance(wav, (str, bytes)):
            wav = [wav]
        
        audio_tensors = []
        for item in wav:
            if isinstance(item, bytes):
                # Load from bytes
                audio_data, orig_sr = sf.read(io.BytesIO(item))
                audio = torch.from_numpy(audio_data).float()
                if audio.ndim == 2:  # stereo -> mono
                    audio = audio.mean(dim=1)
            else:
                # Load from file path
                audio, orig_sr = torchaudio.load(item)
                audio = audio.squeeze(0).float()  # (1, T) -> (T,)
            
            if orig_sr != sr:
                audio = torchaudio.functional.resample(audio.unsqueeze(0), orig_sr, sr).squeeze(0)
            audio_tensors.append(audio)
        
        # Pad to same length and stack
        max_len = max(t.shape[0] for t in audio_tensors)
        padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in audio_tensors]
        audio_batch = torch.stack(padded, dim=0).unsqueeze(1).to(self.device)  # (B, 1, T)
        
        # V2 encoder/decoder API requires prosody features
        enc = self.fa_encoder(audio_batch)
        prosody = self.fa_encoder.get_prosody_feature(audio_batch)
        
        # Align prosody length to match enc length (they can differ by 1 frame)
        if prosody.shape[2] != enc.shape[2]:
            if prosody.shape[2] > enc.shape[2]:
                prosody = prosody[:, :, :enc.shape[2]]
            else:
                pad_len = enc.shape[2] - prosody.shape[2]
                prosody = torch.nn.functional.pad(prosody, (0, pad_len))
        
        vq_pos_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(
            enc, prosody, eval_vq=False, vq=True
        )
        # FACodec V2 ordering (from official test.py):
        # vq_id shape: (num_quantizers, B, T)
        # prosody_code = vq_id[:1]      # (1, B, T)
        # content_code = vq_id[1:3]     # (2, B, T)
        # residual_code = vq_id[3:]     # (3, B, T)
        Qp = vq_id[:1]        # prosody (1, B, T)
        Qc = vq_id[1:3]       # content (2, B, T)
        Qr = vq_id[3:]        # residual (3, B, T)

        # pad or truncate to max_seq_len
        # as per paper, each token corresponds to 12.5ms of audio
        # default max_seq is 1024, which corresponds to ~12.8 seconds of audio
        def pad_or_truncate(q):
            if q.shape[2] > self.max_seq_len:
                return q[:, :, :self.max_seq_len]
            elif q.shape[2] < self.max_seq_len:
                pad_len = self.max_seq_len - q.shape[2]
                return torch.cat([q, torch.zeros(q.shape[0], q.shape[1], pad_len, device=q.device)], dim=2)
            return q
        
        Qp = pad_or_truncate(Qp)  # (1, B, T) prosody
        Qc = pad_or_truncate(Qc)  # (2, B, T) content
        Qr = pad_or_truncate(Qr)  # (3, B, T) residual

        # Style codec = prosody + residuals
        Ys = torch.cat((Qp, Qr), dim=0)  # (4, B, T)
        Yc = Qc                          # (2, B, T)

        # ControlSpeech expects concat(Ys, Yc) along quantizer dimension
        codec = torch.cat((Ys, Yc), dim=0)  # (6, B, T)

        # Project spk_embs from 256 to 512 to match ControlSpeech paper
        # This is ALWAYS done (required by architecture)
        spk_embs = self.style_projection(spk_embs)  # (B, 256) -> (B, 512)

        # Return (B, T, C)
        return codec.permute(1, 2, 0), spk_embs
        # final shape (B (batch), T (seq_len), C (codes))

class AudioEncoder(BaseAudioPreprocessor):
    def __init__(self, dataset, encoder: FACodecEncoder):
        super().__init__()
        self.codec = encoder
        self.dataset = dataset
        self.dir = Path(dataset.root).parent / 'codec_audio'
        self.dir.mkdir(parents=True, exist_ok=True)

    def encode(self, wav: str | list[str]):
        if isinstance(wav, str):
            wav = [wav]

        encoded = [self.codec.encode(w)[0] for w in wav]  # [0] gets codec, not spk_emb
        return torch.stack(encoded, dim=0)
 
if __name__ == '__main__':
    
    test_file = 'test.wav'
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        exit(1)
    
    preprocessor = BaseAudioPreprocessor(sample_rate=16000)
    
    # Load raw audio
    raw_wav, orig_sr = librosa.load(test_file, sr=None)
    print("=== Original Audio ===")
    print(f"  Sample rate: {orig_sr} Hz")
    print(f"  Duration: {len(raw_wav) / orig_sr:.2f}s")
    print(f"  Peak amplitude: {np.abs(raw_wav).max():.4f}")
    
    # Measure loudness
    meter = pyln.Meter(orig_sr)
    orig_loudness = meter.integrated_loudness(raw_wav)
    print(f"  Loudness: {orig_loudness:.2f} LUFS")
    
    # Run preprocessing
    processed_wav, processed_sr = preprocessor.preprocess(test_file)
    
    print("\n=== Preprocessed Audio ===")
    print(f"  Sample rate: {processed_sr} Hz")
    print(f"  Duration: {len(processed_wav) / processed_sr:.2f}s")
    print(f"  Peak amplitude: {np.abs(processed_wav).max():.4f}")
    
    meter = pyln.Meter(processed_sr)
    new_loudness = meter.integrated_loudness(processed_wav)
    print(f"  Loudness: {new_loudness:.2f} LUFS")
    
    # Save preprocessed output
    output_file = 'test_preprocessed.wav'
    preprocessor.save_wav(processed_wav, output_file)
    print(f"\n=== Saved to {output_file} ===")
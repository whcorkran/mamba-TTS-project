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

from transformers import AutoModel, AutoTokenizer
from lib.naturalspeech3_facodec.ns3_codec import FACodecEncoder2, FACodecDecoder2
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
    FACodec pretrained model from
    https://github.com/lifeiteng/naturalspeech3_facodec.git
    """

    def __init__(self, max_seq_len: int = 1024):
        super().__init__()
        self.sequence_length = max_seq_len,
        self.fa_encoder = FACodecEncoder2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.fa_decoder = FACodecDecoder2(
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
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin"
        )
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin"
        )

        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=True))
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt, weights_only=True))

        self.fa_encoder.eval().requires_grad_(False)
        self.fa_decoder.eval().requires_grad_(False)

    def encode(self, wav : str | list[str]):
        # FACodec expects (B, 1, T) or (B, T). We keep (B, 1, T).
        enc = self.fa_encoder(wav)
        vq_pos_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(
            enc, eval_vq=False, vq=True
        )
        # FACodec ordering:
        # vq_id shape: (num_quantizers, B, T)
        # Qc = content, Qp = prosody, Qr = 3 residual/acoustic levels
        Qc = vq_id[0:1]       # (1, B, T)
        Qp = vq_id[1:2]       # (1, B, T)
        Qr = vq_id[2:]        # (3, B, T)

        # pad or truncate to max_seq_len
        # as per paper, each token corresponds to 12.5ms of audio
        # default max_seq is 1600, which corresponds to 20 seconds of audio
        for q in [Qc, Qp, Qr]:
            if q.shape[2] > self.max_seq_len:
                q = q[:, :, :self.max_seq_len]
            elif q.shape[2] < self.max_seq_len:
                q = torch.cat([q, torch.zeros(1, q.shape[1], self.max_seq_len - q.shape[2])], dim=2)

        # Style codec = prosody + residuals
        Ys = torch.cat((Qp, Qr), dim=0)  # (4, B, T)
        Yc = Qc                          # (1, B, T)

        # ControlSpeech expects concat(Ys, Yc) along quantizer dimension
        codec = torch.cat((Ys, Yc), dim=0)  # (5, B, T)

        # Return (B, T, C)
        return codec.permute(1, 2, 0), spk_embs
        # final shape (B (batch), T (seq_len), C (codes))

class AudioEncoder(BaseAudioPreprocessor):
    def __init__(self, dataset ,encoder: FACodecEncoder):
        super().__init__()
        self.codec = encoder
        self.dataset = dataset
        self.dir = Path(dataset.root).parent / 'codec_audio'
        self.dir.mkdir(parents=True, exist_ok=True)

    def encode(self, wav: str | list[str]):
        if isinstance(wav, str):
            wav = [wav]

        encoded = [self.codec.encode(wav) for w in wav]
        torch.stack(encoded, dim=0)
 
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
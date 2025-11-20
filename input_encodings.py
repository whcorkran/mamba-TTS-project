import warnings
warnings.filterwarnings("ignore")

from lib.naturalspeech3_facodec.ns3_codec import FACodecEncoder2, FACodecDecoder2
from huggingface_hub import hf_hub_download
import torchaudio
import torch


class FACodec:
    """
    FACodec pretrained model from
    https://github.com/lifeiteng/naturalspeech3_facodec.git
    """
    def __init__(self):
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
        encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=True))
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt, weights_only=True))

        self.fa_encoder.eval()
        self.fa_decoder.eval()

    def get_latent_Y(self, wav):
        enc = self.fa_encoder(wav)
        vq_pos_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(enc, eval_vq=False, vq=True)

        # concat along channel dim (input must be batched)
        style = torch.cat((vq_id[:1], vq_id[3:]), dim=0)
        codec = torch.cat((style, vq_id[1:3]), dim=0)
        return codec, spk_embs


def test():
    fa_codec = FACodec()
    wav, sr = torchaudio.load("./test.wav")   # wav: (C, T)
    wav = wav.mean(dim=0, keepdim=True)  # ensure mono (1, T)

    wav = wav.unsqueeze(0)   # (1, 1, T)

    # Normalization may be required:
    wav = wav / wav.abs().max().clamp(min=1e-8)

    encoded = fa_codec.get_latent_Y(wav)
    print([e.shape for e in encoded])

if __name__ == "__main__":
    test()
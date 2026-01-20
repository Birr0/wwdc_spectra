import torch.nn as nn
import torch
from aion.codecs import CodecManager
from aion.codecs.config import MODALITY_CODEC_MAPPING

CODEC_CLASS = MODALITY_CODEC_MAPPING[list(MODALITY_CODEC_MAPPING.keys())[39]]
MODALITY_TYPE = list(MODALITY_CODEC_MAPPING.keys())[39]


class PretrainedAIONSpectra(nn.Module):
    # weight init would be a useful option here
    # for full pre-training.
    def __init__(self, reduction="mean_pool"):
        super().__init__()
        codec_manager = CodecManager()
        self.model = codec_manager._load_codec_from_hf(
            codec_class=CODEC_CLASS,
            modality_type=MODALITY_TYPE,  # MODALITY_CODEC_MAPPING.aion.modalities.SDSSSpectrum,
        )
        self.reduction = reduction

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, X):
        code = self.model.encoder.forward(X)
        if self.reduction == "mean_pool":
            return code.mean(dim=-1)
        else:
            msg = "Only Mean Pool (reduction='mean_pool') has been implemented."
            raise NotImplementedError(msg)

    @torch.no_grad()
    def decoder(self, Z):
        return self.model.decoder.forward(Z)


if __name__ == "__main__":
    x = torch.rand(1, 2, 3854)
    model = PretrainedAIONSpectra()
    print(model.encode(x).shape)

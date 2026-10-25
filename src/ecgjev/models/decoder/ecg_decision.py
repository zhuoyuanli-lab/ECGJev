import torch

from ecgjev.models.decoder.startlux_decision.model import IMAGE, StartLuxDecision

ECG_MARK = IMAGE


class ECGDecision(StartLuxDecision):
    def _encode_images(self, images):
        if getattr(self, "vision", None) is None:
            raise ValueError(
                f"ECG injection relies on the checkpoint's multimodal position ids: "
                f"{getattr(self, 'vision_note', 'this backend reads text only')}"
            )
        d_dec = self.body.get_input_embeddings().embedding_dim
        feats = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, d_dec)
            for x in images
        ]
        merge = self.vision.visual.spatial_merge_size
        grid = torch.tensor([[1, merge, f.shape[0] * merge] for f in feats], dtype=torch.long)
        return {
            "features": torch.cat(feats, 0),
            "grid": grid,
            "tokens": [f.shape[0] for f in feats],
        }

    def decide_ecg(self, state, questions, ecg_tokens):
        return self.decide(state, questions, images=[ecg_tokens])

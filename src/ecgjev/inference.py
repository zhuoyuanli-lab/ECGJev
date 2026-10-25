import torch


@torch.no_grad()
def predict_scores(model, signal, visual, rows, **kwargs):
    if model.training:
        raise RuntimeError("call model.eval() before predict()")
    return model(signal, visual, rows, **kwargs).softmax(-1)

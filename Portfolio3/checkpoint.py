import torch
import os


def save(model, path, step, elo):
    os.makedirs(path, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "step": step,
        "elo": elo
    }, f"{path}/ckpt_{step}.pt")


def load_latest(model, path):
    if not os.path.exists(path):
        return model, 0, 1000

    files = sorted(os.listdir(path))
    if not files:
        return model, 0, 1000

    data = torch.load(os.path.join(path, files[-1]))

    model.load_state_dict(data["model"])
    return model, data["step"], data["elo"]
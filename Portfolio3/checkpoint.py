import torch
import os


def save_checkpoint(path, model, optimizer, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step
    }, path)


def load_checkpoint(path, model, optimizer):
    if not os.path.exists(path):
        return 0

    ckpt = torch.load(path)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    print(f"[Checkpoint] Loaded from step {ckpt['step']}")
    return ckpt["step"]
import torch
import os


def save(model, path, step):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/model_{step}.pt")


def load_latest(model, path):
    if not os.path.exists(path):
        return model, 0

    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    if not files:
        return model, 0

    data = torch.load(os.path.join(path, files[-1]))
    model.load_state_dict(data)
    return model, int(files[-1].split("_")[1].split(".")[0])
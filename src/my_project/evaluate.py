#%%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from my_project.data import load_processed_data
from my_project.model import BaselineModel

#%%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")

def evaluate(model_checkpoint: str = MODEL_PATH.joinpath("model.pth")) -> None:
    """Evaluate a trained model."""

    print(model_checkpoint)

    model = BaselineModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = load_processed_data(PROCESSED_DATA_PATH)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")

if __name__ == "__main__":
    typer.run(evaluate)

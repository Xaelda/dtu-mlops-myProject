from pathlib import Path

from tests import _PATH_DATA, _PROJECT_ROOT
from my_project.model import BaselineModel
from my_project.data import load_processed_data
import torch
import typer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 1, print_every: int = 100) -> None:

  # initialize model
  model = BaselineModel().to(DEVICE)

# Load data and create data loader for training data
  train_set, _ = load_processed_data(Path(_PATH_DATA).joinpath("processed")) # TODO make robust in terms of folder input
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

  # define loss function
  criterion = torch.nn.CrossEntropyLoss()

  # define optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        preds, targets = [], []
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            # zero grad -> forward -> loss -> backprop -> update weights (step)
            optimizer.zero_grad()
            y_pred = model.forward(img)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % print_every == 0:
              print(f"Epoch {e}, iter {i}, loss: {loss.item()}")

  print("Training complete")

  torch.save(model.state_dict(), Path(_PROJECT_ROOT).joinpath("models","model.pth"))

if __name__ == "__main__":
  typer.run(train)


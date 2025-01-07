from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from my_project import BaselineModel, load_processed_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

MODEL_PATH = Path("models")
REPORTS_PATH = Path("reports")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5, print_every: int = 100) -> None:
    """Train a model on MNIST."""

    # Print received arguments
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Print every: {print_every}")

    # Define model and print it
    model = BaselineModel().to(DEVICE)
    print(f"\nModel architecture: {model}")

    # Load data and create data loader for training data
    train_set, _ = load_processed_data() # TODO make robust in terms of folder input
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_statistics = {"train_loss": [], "train_accuracy": []}
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # zero grad -> forward -> loss -> backprop -> update weights (step)
            optimizer.zero_grad()
            preds = model.forward(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            training_statistics["train_loss"].append(loss.item())

            accuracy = (preds.argmax(dim=1) == labels).float().mean().item()
            training_statistics["train_accuracy"].append(accuracy)

            if i % print_every == 0:
              print(f"Epoch {e}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), MODEL_PATH.joinpath("model.pth"))

    # Create subfolder if it doesn't exist
    stats_path = REPORTS_PATH.joinpath("statistics")
    # Save statistics to a file
    torch.save(training_statistics, stats_path.joinpath("training_statistics.pth"))

    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(training_statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(training_statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    # Save the figure to the subfolder
    fig.savefig(REPORTS_PATH.joinpath("figures","training_statistics.png"))
    plt.close(fig)


if __name__ == "__main__":
  typer.run(train)

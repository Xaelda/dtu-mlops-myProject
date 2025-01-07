#%%
import matplotlib.pyplot as plt
import torch
import typer
from data import corrupted_mnist
from model import BaselineModel

#%%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# This is a comment that is different from the one in the remote repo
app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5, print_every: int = 100) -> None:
    """Train a model on MNIST."""

    # Print learning rate
    print(f"Received learning rate: {lr}")

    # Define model and print it
    model = BaselineModel().to(DEVICE)
    print(f"\nModel architecture: {model}")

    # Load data and create data loader for training data
    train_set, _ = corrupted_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
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

            statistics["train_loss"].append(loss.item())

            accuracy = (preds.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % print_every == 0:
              print(f"Epoch {e}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""

    print(model_checkpoint)

    model = BaselineModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupted_mnist()
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
    app()






@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = corrupted_mnist()


if __name__ == "__main__":
    app()

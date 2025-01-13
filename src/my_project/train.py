from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
import wandb
from my_project.model import BaselineModel
from my_project.data import load_processed_data
from sklearn.metrics import RocCurveDisplay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
REPORTS_PATH = Path("reports")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5, print_every: int = 100) -> None:
    """Train a model on MNIST."""

    # Print received arguments
    print(f"{lr=}, {batch_size=}, {epochs=}")

    wandb.init(
      project="dtu-mlops-myProject",
      config={"lr": lr, "batch_size": batch_size, "epochs": epochs}
    )

    # Define model and print it
    model = BaselineModel().to(DEVICE)
    print(f"\nModel architecture: {model}")

    # Load data and create data loader for training data
    train_set, _ = load_processed_data(PROCESSED_DATA_PATH) # TODO make robust in terms of folder input
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_statistics = {"train_loss": [], "train_accuracy": []}
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

            training_statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            training_statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(),
                       "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % print_every == 0:
              print(f"Epoch {e}, iter {i}, loss: {loss.item()}")

              # add a plot of the input images
              images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
              wandb.log({"images": images})

              # add a plot of histogram of the gradients
              grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
              wandb.log({"gradients": wandb.Histogram(grads)})

        ## TODO make ROC work
        # # add a custom matplotlib plot of the ROC curves
        # preds = torch.cat(preds, 0)
        # targets = torch.cat(targets, 0)

        # for class_id in range(10):
        #     one_hot = torch.zeros_like(targets)
        #     one_hot[targets == class_id] = 1
        #     _ = RocCurveDisplay.from_predictions(
        #         one_hot,
        #         preds[:, class_id],
        #         name=f"ROC curve for {class_id}",
        #         plot_chance_level=(class_id == 2),
        #     )

        # wandb.plot({"roc": plt})
        # # alternatively the wandb.plot.roc_curve function can be used

    print("Training complete")
    torch.save(model.state_dict(), MODEL_PATH.joinpath("model.pth"))

    # Save statistics to a file
    torch.save(training_statistics, REPORTS_PATH.joinpath("training_statistics.pth"))

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

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from my_project.model import BaselineModel

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")
REPORTS_PATH = Path("reports")

def load_pretrained_model(model_checkpoint: str = MODEL_PATH.joinpath("model.pth")) -> torch.nn.Module:
    """Load a trained model."""
    #model = BaselineModel().to(DEVICE)
    model = BaselineModel()
    model.load_state_dict(torch.load(model_checkpoint))
    return model

def visualize(figure_name: str = "embeddings.png") -> None:
  """ Visualize model features ... TODO write more details"""

  model = load_pretrained_model(MODEL_PATH.joinpath("model.pth"))

  model.eval()
  model.fc = torch.nn.Identity()

  test_images = torch.load(PROCESSED_DATA_PATH.joinpath("test_images.pt"))
  test_targets = torch.load(PROCESSED_DATA_PATH.joinpath("test_target.pt"))
  test_dataset = torch.utils.data.TensorDataset(test_images, test_targets)

  embeddings, targets = [], []
  with torch.inference_mode():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
      images, target = batch
      predictions = model(images)
      embeddings.append(predictions)
      targets.append(target)
    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

  if embeddings.shape[1] > 500: # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
  tsne = TSNE(n_components=2)
  embeddings = tsne.fit_transform(embeddings)

  plt.figure(figsize=(10, 10))
  for i in range(10):
      mask = targets == i
      plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
  plt.legend()
  plt.savefig(REPORTS_PATH.joinpath("figures", figure_name))

if __name__ == "__main__":
    typer.run(visualize)



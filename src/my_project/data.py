from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.raw_data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder.

        The steps taken to preprocess the data are:
        1. Load the raw data.
        2. Concatenate the training images to one tensor. Same with test images.
        3. Indicate expected types of training and test images (torch tensors).
        4. Add an extra dimension to the images tensors.
        5. Normalize the images.
        6. Save the preprocessed data to the output folder.

        """
        # Define empty lists and loop to load all training and test data

        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(self.raw_data_path.joinpath("train_images_{i}.pt")))
            train_target.append(torch.load(self.raw_data_path.joinpath("train_target_{i}.pt")))

        # Concatenate the lists to create a single tensor with torch.cat
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        # Indicate expected type of test_images and test_target
        test_images: torch.Tensor = torch.load(self.raw_data_path.joinpath("test_images.pt"))
        test_target: torch.Tensor = torch.load(self.raw_data_path.joinpath("test_target.pt"))

        # Add another dimension to the images tensors
        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_target = train_target.long()
        test_target = test_target.long()

        train_images = normalize(train_images)
        test_images = normalize(test_images)

        torch.save(train_images, output_folder.joinpath("train_images.pt"))
        torch.save(train_target, output_folder.joinpath("train_target.pt"))
        torch.save(test_images, output_folder.joinpath("test_images.pt"))
        torch.save(test_target, output_folder.joinpath("test_target.pt"))


def preprocess(raw_data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

def load_processed_data(processed_data_path: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """ Load the preprocessed data and return train and test dataloaders.
        The preprocessed data consist of the corrupted MNIST data.
    """
    train_images = torch.load(processed_data_path.joinpath("train_images.pt"))
    train_target = torch.load(processed_data_path.joinpath("train_target.pt"))
    test_images = torch.load(processed_data_path.joinpath("test_images.pt"))
    test_target = torch.load(processed_data_path.joinpath("test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess)

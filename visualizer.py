#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

#%%
# Path to the .pt file
file_path = "./data/corruptedmnist/train_images_0.pt"

# Load the .pt file
data = torch.load(file_path)

# Check the contents of the .pt file
print(f"Data type: {type(data)}")
print(f"Data shape: {data.shape if isinstance(data, torch.Tensor) else 'N/A'}")

#%%
# Visualize the data (assuming it's a batch of MNIST images)
if isinstance(data, torch.Tensor):
    if data.ndim == 4:  # Batch of images (e.g., [batch_size, 1, 28, 28])
        for i in range(min(5, data.shape[0])):  # Display up to 5 images
            image = data[i, 0]  # Select the first channel (grayscale)
            plt.imshow(image.numpy(), cmap="gray")
            plt.title(f"Image {i + 1}")
            plt.axis("off")
            plt.show()
    elif data.ndim == 3:  # Single image (e.g., [1, 28, 28])
        i = np.random.randint(0, data.shape[0] - 1)
        image = data[i]  # Select the first channel
        plt.imshow(image.numpy(), cmap="gray")
        plt.title(f"Training image number {i + 1}")
        plt.axis("off")
        plt.show()
    else:
        print("Unexpected data dimensions. Cannot visualize.")
else:
    print("The .pt file does not contain a tensor.")

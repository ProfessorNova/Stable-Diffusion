import torch
from tqdm import tqdm


def normalize(tensor, mean=None, std=None):
    if mean is None:
        mean = [0.4750, 0.3940, 0.3079]
    if std is None:
        std = [0.2218, 0.2075, 0.2036]
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    tensor = (tensor - mean) / std
    return tensor


def denormalize(tensor, mean=None, std=None):
    if mean is None:
        mean = [0.4750, 0.3940, 0.3079]
    if std is None:
        std = [0.2218, 0.2075, 0.2036]
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)


# Function to compute the mean and standard deviation of the dataset
# The values for the flower dataset are:
# mean=[0.4750, 0.3940, 0.3079]
# std=[0.2218, 0.2075, 0.2036]
def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    for images, _ in tqdm(loader, desc="Computing mean and std"):
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(dim=0)
        std += images.std(2).sum(dim=0)
        total_samples += images.size(0)
    mean /= total_samples
    std /= total_samples
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    return mean, std

import os

import torch
from matplotlib import pyplot as plt
from torch.optim.swa_utils import AveragedModel

from lib.sd_functions import generate
from lib.unet import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define normalization parameters these are computed for the Flowers102 dataset
    # using the compute_mean_std function from lib/utils.py
    mean = [0.4750, 0.3940, 0.3079]
    std = [0.2218, 0.2075, 0.2036]

    # Load the model
    image_shape = (3, 128, 128)
    model = UNet(
        in_shape=image_shape,
        out_shape=image_shape,
        features=[32, 64, 128, 256, 512],
        embedding_dim=32
    ).to(device)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        device=device
    )
    checkpoint_path = "diffusion_model_pretrained.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    # Load the state dict into ema_model
    ema_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ema_model.eval()
    print("Model loaded successfully")

    # Generate images
    num_images = 8  # Number of images to generate
    diffusion_steps = 20  # Number of steps to reverse the diffusion process (higher is better but 20 should be enough)

    generated_images = generate(
        ema_model,
        num_images,
        diffusion_steps,
        image_shape,
        mean,
        std,
        device
    )

    # Plot the final generated images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle("Generated Images", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    main()

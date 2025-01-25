import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.swa_utils import AveragedModel
from torchvision.datasets import Flowers102
from tqdm import tqdm

from lib.sd_functions import cosine_diffusion_schedule, denoise, generate
from lib.unet import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define normalization parameters these are computed for the Flowers102 dataset
    # using the compute_mean_std function from lib/utils.py
    mean = [0.4750, 0.3940, 0.3079]
    std = [0.2218, 0.2075, 0.2036]

    # Define transformations
    transform = transforms.Compose([
        transforms.PILToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Resize(128),  # Resize the image to have a minimum size of 128x128 pixels
        transforms.CenterCrop(128),  # Crop the image to 128x128 pixels around the center
        transforms.ConvertImageDtype(torch.float32),  # Convert the image to float32
        transforms.Normalize(mean, std)  # Normalize so that data has zero mean and unit variance
    ])

    # Load datasets
    train_data = Flowers102(
        root="data",
        split="train",
        download=True,
        transform=transform
    )

    val_data = Flowers102(
        root="data",
        split="val",
        download=True,
        transform=transform
    )

    test_data = Flowers102(
        root="data",
        split="test",
        download=True,
        transform=transform
    )

    # Combine datasets
    flower_data = torch.utils.data.ConcatDataset([train_data, val_data, test_data])

    # Create DataLoader with multiple workers
    flower_loader = torch.utils.data.DataLoader(
        flower_data,
        batch_size=64,
        shuffle=True,
        num_workers=4,  # Adjust based on the number of CPU cores
        pin_memory=True if str(device) == "cuda" else False,  # Copy data to CUDA pinned memory
        prefetch_factor=2,  # Number of batches loaded in advance by each worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    # Get the shape of one input image
    dummy_input = next(iter(flower_loader))[0][0]

    def train_diffusion_model(model, ema_model, loader, optimizer, device, epochs=50, output_dir="images"):
        os.makedirs(output_dir, exist_ok=True)
        criterion = nn.L1Loss()

        # Use gradient scaling to prevent underflow
        scaler = torch.amp.GradScaler(str(device))

        for epoch in range(1, epochs + 1):
            avg_loss = 0.0
            model.train()
            for images, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
                images = images.to(device, non_blocking=True)
                noises = torch.randn(images.shape, device=device)

                # Sample a random timestep for each image in range [0, 1]
                diffusion_times = torch.rand(images.shape[0], 1, 1, 1, device=device)
                noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times, device)
                # Mix the images with the noise
                noisy_images = signal_rates * images + noise_rates * noises

                optimizer.zero_grad()
                # Forward Pass mit autocast
                with torch.amp.autocast(str(device)):
                    predicted_noises, _ = denoise(model, noisy_images, noise_rates, signal_rates)
                    loss = criterion(noises, predicted_noises)

                # Backward Pass und Optimierung mit GradScaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Aktualisieren Sie die EMA-Parameter
                ema_model.update_parameters(model)

                avg_loss += loss.item()

            avg_loss /= len(loader)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
            scheduler.step(avg_loss)

            # Sample images and save them as a grid
            with torch.no_grad():
                generated_images = generate(ema_model, 8, 50, dummy_input.shape, mean, std, device)
                torchvision.utils.save_image(
                    generated_images, f"{output_dir}/generated_images_{epoch}.png", nrow=4
                )

            # Save the model
            torch.save(ema_model.state_dict(), "diffusion_model.pth")

    # Initialize models, optimizer, and EMA
    model = UNet(
        in_shape=dummy_input.shape,
        out_shape=dummy_input.shape,
        features=[32, 64, 128, 256, 512],
        embedding_dim=32
    ).to(device)

    # Use EMA for model so that the model is more robust to noise
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        device=device
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    # Start training
    train_diffusion_model(model, ema_model, flower_loader, optimizer, device, epochs=1000)


if __name__ == "__main__":
    main()

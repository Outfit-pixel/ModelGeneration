import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import cv2
from torchvision import transforms
from PIL import Image
import lpips 
import torch.nn.functional as F
from models.vqvae import Encoder, VectorQuantizer, Decoder
from config import current_dir, image_folder, model_checkpoints, VQVAE_CONFIG, TRAIN_CONFIG, VQVAE_ENCODER_CONFIG,VQVAE_QUANTIZER_CONFIG

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_filenames = sorted(os.listdir(image_folder))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image

def train_vqvae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    encoder = Encoder(**VQVAE_ENCODER_CONFIG).to(device)
    quantizer = VectorQuantizer(**VQVAE_QUANTIZER_CONFIG).to(device)
    decoder = Decoder(**VQVAE_CONFIG).to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Optimizer and scheduler
    optimizer = Adam(
        list(encoder.parameters()) + 
        list(quantizer.parameters()) + 
        list(decoder.parameters()), 
        lr=TRAIN_CONFIG["learning_rate"]
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    # Dataset and dataloaders
    dataset = ImageDataset(image_folder)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        # Training phase
        encoder.train()
        quantizer.train()
        decoder.train()
        
        train_loss = 0.0
        train_recon_loss = 0.0
        train_vq_loss = 0.0
        train_perceptual_loss = 0.0

        for images in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} - Training"):
            images = images.to(device)
            
            with autocast():
                encoded = encoder(images)
                quantized, vq_loss, _ = quantizer(encoded)
                reconstructed = decoder(quantized)

                recon_loss = F.mse_loss(reconstructed, images)
                perceptual_loss = lpips_loss_fn(reconstructed, images).mean()
                loss = recon_loss + vq_loss + perceptual_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            train_perceptual_loss += perceptual_loss.item()

        # Validation phase
        encoder.eval()
        quantizer.eval()
        decoder.eval()
        
        val_loss = 0.0
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        val_perceptual_loss = 0.0

        with torch.no_grad():
            for images in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} - Validation"):
                images = images.to(device)
                
                with autocast():
                    encoded = encoder(images)
                    quantized, vq_loss, _ = quantizer(encoded)
                    reconstructed = decoder(quantized)

                    recon_loss = F.mse_loss(reconstructed, images)
                    perceptual_loss = lpips_loss_fn(reconstructed, images).mean()
                    loss = recon_loss + vq_loss + perceptual_loss

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()
                val_perceptual_loss += perceptual_loss.item()

        # Print epoch results
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} Results:")
        print(f"  Train Loss: {train_loss/len(train_dataloader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_dataloader):.4f}")

        scheduler.step()

    # Save models
    os.makedirs(model_checkpoints, exist_ok=True)
    torch.save(encoder.state_dict(), model_checkpoints / "final_encoder.pth")
    torch.save(quantizer.state_dict(), model_checkpoints / "final_quantizer.pth")
    torch.save(decoder.state_dict(), model_checkpoints / "final_decoder.pth")
    print("Training completed. Models saved successfully!")

if __name__ == "__main__":
    train_vqvae()
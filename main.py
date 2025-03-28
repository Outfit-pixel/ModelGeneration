from data_preprocessing.generate_captions import generate_captions
from data_preprocessing.generate_text_embeddings import generate_text_embeddings
from training.train_vqvae import train_vqvae
from training.train_transformer import train_transformer
from utils import clear_memory, save_encoding_indices
from models.vqvae import Encoder, VectorQuantizer
from config import current_dir, image_folder, device, VQVAE_CONFIG,VQVAE_ENCODER_CONFIG,VQVAE_QUANTIZER_CONFIG
import torch
def main():
    # Step 1: Generate captions for images
    print("Step 1: Generating captions...")
    generate_captions()
    
    # Step 2: Generate text embeddings
    print("\nStep 2: Generating text embeddings...")
    generate_text_embeddings()
    clear_memory()
    
    # Step 3: Train VQ-VAE
    print("\nStep 3: Training VQ-VAE...")
    train_vqvae()
    clear_memory()
    
    # Step 4: Generate encoding indices
    print("\nStep 4: Generating encoding indices...")
    encoder = Encoder(**VQVAE_ENCODER_CONFIG).to(device)
    quantizer = VectorQuantizer(**VQVAE_QUANTIZER_CONFIG).to(device)
    encoder.load_state_dict(torch.load(current_dir / "model_checkpoints"/ "final_encoder.pth"))
    quantizer.load_state_dict(torch.load(current_dir / "model_checkpoints"/ "final_quantizer.pth"))
    save_encoding_indices(image_folder, encoder, quantizer, device, current_dir)
    clear_memory()
    
    # Step 5: Train Transformer
    print("\nStep 5: Training Transformer...")
    train_transformer()
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
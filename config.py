import torch
import pathlib

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Path configurations
current_dir = pathlib.Path("/content/drive/MyDrive/Model_Generation-anurag/Complete/ModelGeneration")
image_folder = pathlib.Path("/content/drive/MyDrive/Model_Generation-anurag/Complete/ModelGeneration/model_data")
model_checkpoints = current_dir / "model_checkpoints"

VQVAE_ENCODER_CONFIG = {
    "ch": 128,
    "ch_mult": (1, 2, 4, 8),
    "num_res_blocks": 2,
    "in_channels": 3,
    "z_channels": 32
}

VQVAE_QUANTIZER_CONFIG = {
    "num_embeddings": 1024,
    "embedding_dim": 32,
    "beta": 0.25,
    "decay": 0.99
}


# Model parameters
VQVAE_CONFIG = {
    "ch": 128,
    "ch_mult": (1, 2, 4, 8),
    "num_res_blocks": 2,
    "in_channels": 32,
    "out_channels": 3
}

TRANSFORMER_CONFIG = {
    "vocab_size": 8192,
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 6
}

# Training parameters
TRAIN_CONFIG = {
    "batch_size": 2,
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "progressive_steps": [4, 8, 16, 32, 64, 1024]
}
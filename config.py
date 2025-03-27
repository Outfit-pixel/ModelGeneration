import pathlib

# Path configurations
current_dir = pathlib.Path("/kaggle/working/")
image_folder = pathlib.Path("/kaggle/input/deepfashion-1/datasets/train_images")
model_checkpoints = current_dir / "model_checkpoints"

# Model parameters
VQVAE_CONFIG = {
    "ch": 128,
    "ch_mult": (1, 2, 4, 8),
    "num_res_blocks": 2,
    "in_channels": 3,
    "z_channels": 32,
    "num_embeddings": 1024,
    "embedding_dim": 32,
    "beta": 0.25,
    "decay": 0.99
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
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from models.transformer import ConditionalAutoregressiveTransformer
from config import current_dir, model_checkpoints, TRANSFORMER_CONFIG, TRAIN_CONFIG

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, image_tokens, text_embeddings):
        self.image_tokens = image_tokens
        self.text_embeddings = text_embeddings

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        return self.text_embeddings[idx], self.image_tokens[idx]

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    image_tokens = torch.load(current_dir / "final_encoding_indices.pt", map_location=device)
    text_embeddings = torch.load(current_dir / "text_embeddings.pt", map_location=device)
    
    # Create dataset and dataloader
    dataset = ImageTextDataset(image_tokens, text_embeddings)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    model = ConditionalAutoregressiveTransformer(**TRANSFORMER_CONFIG)
    model.to(device)
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    # Training loop
    num_stages = len(TRAIN_CONFIG["progressive_steps"])
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        total_loss = 0
        prog_stage = min(epoch // (TRAIN_CONFIG["num_epochs"] // num_stages), num_stages - 1)
        prog_si = TRAIN_CONFIG["progressive_steps"][prog_stage]
        
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}: Training with {prog_si} tokens")
        
        for text_embeds, image_tokens in train_loader:
            text_embeds, image_tokens = text_embeds.to(device), image_tokens.to(device)
            image_tokens = image_tokens.long()
            
            if image_tokens.shape[1] < prog_si:
                continue
                
            optimizer.zero_grad()
            
            with autocast():
                output = model(image_tokens[:, :prog_si-1], text_embeds)
                output = output.reshape(-1, TRANSFORMER_CONFIG["vocab_size"])
                target = image_tokens[:, 1:prog_si].reshape(-1)
                
                if output.shape[0] != target.shape[0]:
                    min_len = min(output.shape[0], target.shape[0])
                    output = output[:min_len]
                    target = target[:min_len]
                
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_checkpoints / "conditional_autoregressive_transformer.pth")
    print("✅ Transformer model saved successfully!")

if __name__ == "__main__":
    train_transformer()
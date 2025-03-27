import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import current_dir

def generate_text_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('clip-ViT-L-14').to(device)
    
    json_file = current_dir / "captions.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    captions = [item["caption"] for item in data]
    text_embeddings = model.encode(captions, convert_to_tensor=True, device=device)
    
    torch.save(text_embeddings, current_dir / "text_embeddings.pt")
    print(f"✅ Generated and saved {len(text_embeddings)} text embeddings!")

if __name__ == "__main__":
    generate_text_embeddings()
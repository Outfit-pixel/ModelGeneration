import torch
import gc

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def save_encoding_indices(image_folder, encoder, quantizer, device, output_path):
    """Generate and save encoding indices for images"""
    from torchvision import transforms
    from PIL import Image
    import os
    from tqdm import tqdm
    
    encoder.eval()
    quantizer.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image_filenames = sorted(os.listdir(image_folder))
    embeddings = []
    
    with torch.no_grad():
        for img_name in tqdm(image_filenames, desc="Processing Images"):
            img_path = os.path.join(image_folder, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            
            encoded = encoder(image)
            _, _, encoded_indices = quantizer(encoded)
            embeddings.append(encoded_indices.squeeze(0).cpu())
    
    torch.save(embeddings, output_path / "encoding_indices.pt")
    
    # Restructure embeddings
    data = torch.load(output_path / "encoding_indices.pt")
    data = torch.stack(data)
    data = torch.stack([d.view(32, 32) for d in data])
    torch.save(data, output_path / "final_encoding_indices.pt")
    
    print("✅ Image embeddings saved successfully!")
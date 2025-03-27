import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from config import current_dir, image_folder

def generate_captions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)

    output_json = current_dir / "captions.json"
    image_files = sorted(image_folder.glob("*"))
    captions_list = []

    for image_file in tqdm(image_files, desc="Generating Captions"):
        try:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error captioning {image_file.name}: {e}")
            caption = "no caption"

        captions_list.append({"image": image_file.name, "caption": caption})

    with open(output_json, "w") as json_file:
        json.dump(captions_list, json_file, indent=4)

    print(f"Captions saved to {output_json}")

if __name__ == "__main__":
    generate_captions()
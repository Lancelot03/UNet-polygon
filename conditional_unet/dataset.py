

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PolygonColoringDataset(Dataset):
    """
    This version ACCEPTS a pre-built color vocabulary instead of creating its own.
    """
    def __init__(self, root_dir, color_vocab, image_size=128):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        with open(self.root_dir / "data.json", "r") as f:
            self.items = json.load(f)
          
        self.color_to_idx = color_vocab
        self.idx_to_color = {i: color for color, i in self.color_to_idx.items()}
        self.num_colors = len(self.color_to_idx)

        print(f"Dataset for '{self.root_dir.name}' initialized using provided vocab. Found {len(self.items)} samples.")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        metadata = self.items[idx]
        
        input_filename = metadata['input_polygon']
        output_filename = metadata['output_image']
        color_name = metadata['colour']
        
        input_img_path = self.root_dir / "inputs" / input_filename
        output_img_path = self.root_dir / "outputs" / output_filename
        
        input_image = Image.open(input_img_path).convert("RGB")
        output_image = Image.open(output_img_path).convert("RGB")

        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)
        
        color_idx = torch.tensor(self.color_to_idx[color_name], dtype=torch.long)
        
        return {
            "input_image": input_tensor,
            "color_idx": color_idx,
            "target_image": output_tensor
        }

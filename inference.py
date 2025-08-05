
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("Step 1: Libraries imported.")


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_colors, color_embed_dim, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.color_embedding = nn.Embedding(num_colors, color_embed_dim)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cond_mlp1 = nn.Linear(color_embed_dim, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.cond_mlp2 = nn.Linear(color_embed_dim, 256 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.cond_mlp3 = nn.Linear(color_embed_dim, 128 // factor)
        self.up4 = Up(128, 64, bilinear)
        self.cond_mlp4 = nn.Linear(color_embed_dim, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, color_idx):
        c = self.color_embedding(color_idx)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        cond_emb = self.cond_mlp1(c).unsqueeze(-1).unsqueeze(-1)
        x = x + cond_emb
        x = self.up2(x, x3)
        cond_emb = self.cond_mlp2(c).unsqueeze(-1).unsqueeze(-1)
        x = x + cond_emb
        x = self.up3(x, x2)
        cond_emb = self.cond_mlp3(c).unsqueeze(-1).unsqueeze(-1)
        x = x + cond_emb
        x = self.up4(x, x1)
        cond_emb = self.cond_mlp4(c).unsqueeze(-1).unsqueeze(-1)
        x = x + cond_emb
        logits = self.outc(x)
        return torch.sigmoid(logits)

print("Step 2: Model architecture defined.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
COLOR_EMBED_DIM = 32 # This must match the hyperparameter from training.

COLOR_VOCAB = {
    'blue': 0, 'cyan': 1, 'green': 2, 'magenta': 3, 'orange': 4, 
    'purple': 5, 'red': 6, 'yellow': 7
}
NUM_COLORS = len(COLOR_VOCAB)

print("Step 3: Inference configuration set.")
print(f"Using device: {device}")
print(f"Master Vocabulary: {COLOR_VOCAB}")

model = ConditionalUNet(
    n_channels=3, 
    n_classes=3, 
    num_colors=NUM_COLORS, 
    color_embed_dim=COLOR_EMBED_DIM
).to(device)


model_path = "best_unet_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode
    print("\nStep 4: Model loaded successfully from 'best_unet_model.pth'!")
except FileNotFoundError:
    print(f"\nERROR: Model file not found at '{model_path}'.")
    print("Please make sure you have run the training script first and the file was saved correctly.")
except Exception as e:
    print(f"\nERROR: An error occurred while loading the model: {e}")

def predict(model, image_path, color_name, device):
    """Generates a colored polygon from an input image and color name."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device) # Add batch dimension

    if color_name not in COLOR_VOCAB:
        raise ValueError(f"Color '{color_name}' not in vocabulary. Available colors: {list(COLOR_VOCAB.keys())}")
    
    color_idx = torch.tensor([COLOR_VOCAB[color_name]], dtype=torch.long).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor, color_idx)

    output_tensor = output_tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_tensor)
    
    return input_image, output_image

print("Step 5: Prediction function defined.")

def show_prediction(image_path, color_name):
    """A helper function to run prediction and plot the results."""
    try:
        input_img, generated_img = predict(model, image_path, color_name, device)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display Input Image
        axes[0].imshow(input_img.resize((256, 256))) # Show a larger version for clarity
        axes[0].set_title(f"Input: {Path(image_path).name}")
        axes[0].axis('off')
        
        # Display Generated Output
        axes[1].imshow(generated_img)
        axes[1].set_title(f"Generated Output (Color: {color_name})")
        axes[1].axis('off')
        
        plt.suptitle(f"UNet Polygon Coloring Inference", fontsize=16)
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Input file not found at '{image_path}'")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

print("\nStep 6: Running inference on validation and training examples...")

print("\n--- Validation Set Examples ---")
show_prediction("dataset/validation/inputs/triangle.png", "red")
show_prediction("dataset/validation/inputs/square.png", "blue")
show_prediction("dataset/validation/inputs/octagon.png", "yellow")
show_prediction("dataset/validation/inputs/heptagon.png", "green")
show_prediction("dataset/validation/inputs/pentagon.png", "magenta")

print("\n--- Training Set Examples (for more variety) ---")
show_prediction("dataset/training/inputs/hexagon.png", "cyan")
show_prediction("dataset/training/inputs/nonagon.png", "purple")
show_prediction("dataset/training/inputs/decagon.png", "orange")

# train.py (Builds a master vocab and passes it down)

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from conditional_unet.dataset import PolygonColoringDataset
from conditional_unet.model import ConditionalUNet
from conditional_unet.utils import log_images_to_wandb

def train(config):
    wandb.init(project="polygon-coloring-unet", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CHANGE: Build a single, unified color vocabulary ---
    print("Building master color vocabulary...")
    with open("dataset/training/data.json", 'r') as f:
        train_data = json.load(f)
    with open("dataset/validation/data.json", 'r') as f:
        val_data = json.load(f)
    
    all_colors = set(item['colour'] for item in train_data)
    all_colors.update(item['colour'] for item in val_data)
    
    master_color_vocab = {color: i for i, color in enumerate(sorted(list(all_colors)))}
    
    print("Master Vocabulary Created:")
    print(master_color_vocab)
    print("-" * 30)

    # --- CHANGE: Pass the master vocab to both datasets ---
    train_dataset = PolygonColoringDataset(
        root_dir="dataset/training", 
        color_vocab=master_color_vocab, 
        image_size=config.image_size
    )
    val_dataset = PolygonColoringDataset(
        root_dir="dataset/validation", 
        color_vocab=master_color_vocab, 
        image_size=config.image_size
    )
    
    # This assertion should now pass, but we keep it as a sanity check.
    assert train_dataset.color_to_idx == val_dataset.color_to_idx, "Vocabularies still don't match!"
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # --- CHANGE: Use the size of the master vocab for the model ---
    model = ConditionalUNet(
        n_channels=3, 
        n_classes=3, 
        num_colors=len(master_color_vocab), 
        color_embed_dim=config.color_embed_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in progress_bar:
            input_images = batch['input_image'].to(device)
            color_indices = batch['color_idx'].to(device)
            target_images = batch['target_image'].to(device)
            outputs = model(input_images, color_indices)
            loss = criterion(outputs, target_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_images = batch['input_image'].to(device)
                color_indices = batch['color_idx'].to(device)
                target_images = batch['target_image'].to(device)
                outputs = model(input_images, color_indices)
                loss = criterion(outputs, target_images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        log_images_to_wandb(model, val_loader, device, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            wandb.save("best_unet_model.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Conditional UNet for polygon coloring.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--image_size", type=int, default=128, help="Size to resize images to.")
    parser.add_argument("--color_embed_dim", type=int, default=32, help="Dimension of the color embedding.")
    
    args = parser.parse_args()
    train(args)

# conditional_unet/utils.py

import torch
import wandb
from torchvision.utils import make_grid

def log_images_to_wandb(model, dataloader, device, epoch, num_images=5):
    """Logs a grid of input, prediction, and ground truth images to W&B."""
    model.eval()
    images, predictions, ground_truths = [], [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(images) >= num_images:
                break
            
            input_img = batch['input_image'].to(device)
            color_idx = batch['color_idx'].to(device)
            target_img = batch['target_image'].to(device)
            
            pred_img = model(input_img, color_idx)
            
            # Add one image from the batch to the lists
            images.append(input_img[0].cpu())
            predictions.append(pred_img[0].cpu())
            ground_truths.append(target_img[0].cpu())

    # Create grids
    # Concatenate [input, prediction, ground_truth] for each example
    combined_images = []
    for i, p, gt in zip(images, predictions, ground_truths):
        combined_images.extend([i, p, gt])
        
    grid = make_grid(combined_images, nrow=3) # 3 columns: input, pred, gt
    
    wandb.log({
        "epoch": epoch,
        "validation_examples": wandb.Image(grid, caption="Rows: [Input, Prediction, Ground Truth]")
    })
    model.train()

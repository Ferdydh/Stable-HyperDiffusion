import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch

def plot_n_images(images, row=False):
    """
    Plot multiple images in a grid or single-row layout.

    :param images: List of images to plot (numpy arrays or torch tensors).
    :param row: If True, display all images in a single row; otherwise, in a grid.
    """

    n_samples = len(images)

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(n_samples))  # Number of columns
    rows = math.ceil(n_samples / cols)

    # Distinguish between grid or row layout
    if row:
        plt.figure(figsize=(15, 5))

        with torch.no_grad():
            for i in range(n_samples):
                plt.subplot(1, n_samples, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.axis('off')
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

        # Flatten axes array for easy indexing
        axes = axes.ravel()

        for i in range(n_samples):
            axes[i].imshow(images[i], cmap="gray", extent=(-1, 1, -1, 1))
            axes[i].axis("off")

        for j in range(n_samples, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()  
    plt.ion()
    plt.show(block=True)


def plot_n_images_and_reconstructions(images, reconstructions, mse_weights, mse_images):
    """
    Plot original and reconstructed along with MSE values in grid layout.

    :param images: List of original images.
    :param reconstructions: List of reconstructed images.
    :param mse_weights: List of weight MSE values.
    :param mse_images: List of image MSE values.
    """

    n_samples = len(images)

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(n_samples*2))  # Number of columns
    rows = 2*math.ceil(n_samples / cols)
    cols = 2*cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(n_samples):
        col = (2*i) % cols
        row = 2*((2*i) // cols)

        # Plot original image
        axes[row, col].imshow(images[i], cmap="gray", extent=(-1, 1, -1, 1))
        axes[row, col].axis("off")
        axes[row, col].set_title("Original", fontsize=8)

        # Plot reconstruction
        axes[row+1,col].imshow(reconstructions[i], cmap="gray", extent=(-1, 1, -1, 1))
        axes[row+1,col].axis("off")
        axes[row+1,col].set_title("Reconstruction", fontsize=8)

        # Add mse values
        axes[row+1, col+1].set_title(f"Image MSE: {mse_images[i]:.4f}\nWeight MSE: {mse_weights[i]:.4f}", fontsize=8)

    # Flatten axes array for easy indexing
    axes = axes.ravel()

    # Delete empty subplots
    for j in range(len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  
    plt.ion()
    plt.show(block=True)


def render_n_images_and_reconstructions(images, reconstructions, mse_weights, mse_images, path, spacing=20, in_sample_spacing=10, boundary_spacing=10, scale_factor=5):
    """
    Renders images and reconstructions in grid layout and saves them to a file.
    """
    n = len(images)
    
    # Calculate optimal grid layout
    cols = math.ceil(math.sqrt(2*n))
    rows = math.ceil(n / cols)

    image_size = images[0].shape[0] * scale_factor
    spacing *= scale_factor
    in_sample_spacing *= scale_factor
    boundary_spacing *= scale_factor

    
    width = cols * image_size + boundary_spacing * 2 +  (cols - 1) * spacing
    height = rows * (2*image_size + in_sample_spacing) + boundary_spacing * 2 + (rows - 1) * spacing

    canvas = Image.new('L', (width, height), color=255)
    
    # Choose a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(canvas)

    for idx in range(n):
        row = idx // cols
        col = idx % cols
        
        # Create nested GridSpec for this pair of images
        #inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Get the images
        orig_img = images[idx]
        recon_img = reconstructions[idx]
        weight_mse = mse_weights[idx]
        image_mse = mse_images[idx]
        
        # Convert from torch tensor if necessary
        if isinstance(orig_img, torch.Tensor):
            orig_img = orig_img.cpu().detach().numpy()
        if isinstance(recon_img, torch.Tensor):
            recon_img = recon_img.cpu().detach().numpy()
        
        orig_img = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255).astype(np.uint8)
        orig_img = np.repeat(orig_img, scale_factor, axis=0)
        orig_img = np.repeat(orig_img, scale_factor, axis=1)
        recon_img = ((recon_img - recon_img.min()) / (recon_img.max() - recon_img.min()) * 255).astype(np.uint8)
        recon_img = np.repeat(recon_img, scale_factor, axis=0)
        recon_img = np.repeat(recon_img, scale_factor, axis=1)
        
        x_pos = boundary_spacing + col * (image_size + spacing)
        y_pos = boundary_spacing + row * (2*image_size + in_sample_spacing + spacing)

        canvas.paste(Image.fromarray(orig_img, mode='L'), (x_pos, y_pos))
        canvas.paste(Image.fromarray(recon_img, mode='L'), (x_pos, y_pos + image_size + in_sample_spacing))

        image_mse_label = f"Image mse: {image_mse:.6f}"
        text_x = x_pos + 5
        text_y = y_pos + image_size + 5 
        
        # Draw text
        draw.text((text_x, text_y), image_mse_label, fill='black', font=font)
        weight_mse_label = f"Weight mse: {weight_mse:.6f}"
        text_x = x_pos + 5
        text_y = y_pos + image_size + 20
        draw.text((text_x, text_y), weight_mse_label, fill='black', font=font)

    # Save the canvas to image file
    canvas.save(path, format='PNG')
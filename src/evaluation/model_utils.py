import numpy as np
import torch
from src.data.data_converter import tokens_to_weights
from src.evaluation.metrics import calculate_mse
from src.data.data_converter import weights_to_flattened_weights

def compute_image(inr, state_dict):
    """
    Generate a 2D image by evaluating the INR model.

    :param inr: Implicit Neural Representation (INR) model.
    :param state_dict: State dictionary containing model weights.
    :param resolution: Resolution of the generated image (default=28x28).
    :return: Generated image as a numpy array.
    """
    inr.eval()
    inr.load_state_dict(state_dict)
    resolution = 28

    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    grid_x, grid_y = np.meshgrid(x, y)

    inputs = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        outputs = inr(inputs_tensor).cpu().numpy()

    image = outputs.reshape(resolution, resolution)

    return image


def get_best_samples(images, recons, mse_weights, image_mses, best_n=10, sort_by="weight"):
    """
    Select the best samples based on MSE.

    :param images: List of original images.
    :param recons: List of reconstructed images.
    :param mse_weights: List of weight MSE values.
    :param image_mses: List of image MSE values.
    :param best_n: Number of best samples to select.
    :param sort_by: Criterion to sort by ('weight' or 'image').
    :return: Top-N best samples as a list of tuples.
    """
    if sort_by == "weight":
        index = 2
    elif sort_by == "image":
        index = 3
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}. Must be 'weight' or 'image'.")
    return sorted(zip(images, recons, mse_weights, image_mses), key=lambda x: x[index])[:best_n]


def sample_from_latent_space(vae, inr, ref_cp, pos, n_tokens, latent_dim, n_samples=10, mean=None, std=None):
    """
    Samples from the latent space and generates images.
    """
    if pos.shape[0] != n_samples:
        if pos.ndim > 3 or pos.ndim < 2 or (pos.shape[0] != n_tokens and pos.shape[1] != n_tokens):
            raise ValueError(f"pos must be a 2D or 3D tensor with shape ({n_tokens}, 3) or ({n_samples}, {n_tokens}, 3), got shape {pos.shape}")
        pos = pos.unsqueeze(0).repeat(n_samples, 1, 1)

    input_samples = torch.randn(n_samples, n_tokens, latent_dim)

    if mean and std:
        input_samples = mean + std * input_samples
    
    vae.eval()
    with torch.no_grad():
        output = vae.decoder(input_samples, pos)

    images = [
        compute_image(inr, tokens_to_weights(t, p, ref_cp)) for t, p in zip(output, pos)
    ]

    return images


# Function that interpolates between two latent vectors and generates the respective images
def interpolate_latent_space(latent_vector0, latent_vector1, num_interpolation_steps, pos, vae, ref_cp, inr):
    """
    Takes two latent vectors, interpolates between them and outputs the respective reconstructions
    """
    alpha_values = np.linspace(0, 1, num_interpolation_steps)
    latents = []
    positions = []
    # Interpolate latent space for image mse
    for alpha in alpha_values:
        interpolated_latent = (1 - alpha) * latent_vector0 + alpha * latent_vector1
        latents.append(interpolated_latent)
        positions.append(pos)

    latents = torch.stack(latents)
    positions = torch.stack(positions)

    # Decode the interpolated latent space
    vae.eval()
    with torch.no_grad():
        decoded_latens = vae.decoder(latents, positions)
    
    images = [
        compute_image(inr, tokens_to_weights(t, p, ref_cp)) for t, p in zip(decoded_latens, positions)
    ]

    return images



def get_n_images_and_mses(vae, dataset, inr, n_samples, random=True):
    if random:
        idx = np.random.choice(len(dataset), n_samples, replace=False)
    else:
        #if n_samples is None or n_samples > len(dataset):
        #    n_samples = len(dataset)

        #if n_samples > 128:
        #    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
        #else:
        #    model_input = torch.stack([dataset[i][0] for i in range(n_samples)])
        #    positions = dataset[0][2].unsqueeze(0).repeat(n_samples, 1, 1)
        idx = np.arange(n_samples)

    positions = dataset[0][2].unsqueeze(0).repeat(n_samples, 1, 1)

    ref_cp = dataset.get_state_dict(index=0)

    inputs = torch.stack([dataset[i][0] for i in idx])

    vae.eval()
    with torch.no_grad():
        outputs,_,_,_ = vae(inputs, positions)
    state_dicts_output = [tokens_to_weights(t, p, ref_cp) for t, p in zip(outputs, positions)]
    state_dicts_input = [tokens_to_weights(t, p, ref_cp) for t, p in zip(inputs, positions)]

    images = []
    recons = []
    mse_weights = []
    mse_images = []
    for i in range(n_samples):
        state_dict_input = state_dicts_input[i]
        state_dict_output = state_dicts_output[i]

        image = compute_image(inr, state_dict_input)
        image_recon = compute_image(inr, state_dict_output)

        images.append(image)
        recons.append(image_recon)

        mse_images.append(calculate_mse(image, image_recon))

        flattened_weights_orig = weights_to_flattened_weights(state_dict_input)
        flattened_weights_recon = weights_to_flattened_weights(state_dict_output)

        mse_weights.append(calculate_mse(flattened_weights_orig, flattened_weights_recon))
    return images, recons, mse_weights, mse_images
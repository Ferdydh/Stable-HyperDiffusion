import numpy as np
import torch
from src.data.data_converter import tokens_to_weights
from src.data.data_converter import weights_to_flattened_weights, flattened_weights_to_weights
from src.evaluation.metrics import calculate_mse, find_nearest_neighbor


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

    :param vae: Variational Autoencoder (VAE) model.
    :param inr: Implicit Neural Representation (INR) model.
    :param ref_cp: Reference checkpoint for the model.
    :param pos: Tensor of positional encodings.
    :param n_tokens: Number of tokens for input.
    :param latent_dim: Dimensionality of the latent space.
    :param n_samples: Number of samples to generate.
    :param mean: Optional mean for sampling.
    :param std: Optional standard deviation for sampling.
    :return: List of generated images.
    """
    if pos.shape[0] != n_samples:
        if pos.ndim > 3 or pos.ndim < 2 or (pos.shape[0] != n_tokens and pos.shape[1] != n_tokens):
            raise ValueError(f"pos must be a 2D or 3D tensor with shape ({n_tokens}, 3) or ({n_samples}, {n_tokens}, 3), got shape {pos.shape}")
        pos = pos.unsqueeze(0).repeat(n_samples, 1, 1)

    # Sample random points in latent space.
    input_samples = torch.randn(n_samples, n_tokens, latent_dim)

    if mean and std:
        input_samples = mean + std * input_samples

    # Decode the latent samples to generate tokens.
    vae.eval()
    with torch.no_grad():
        output = vae.decoder(input_samples, pos)

    # Generate images for each token set.
    images = [
        compute_image(inr, tokens_to_weights(t, p, ref_cp)) for t, p in zip(output, pos)
    ]

    return images


def interpolate_latent_space(latent_vector0, latent_vector1, num_interpolation_steps, pos, vae, ref_cp, inr):
    """
    Interpolates between two latent vectors and generates the respective images.

    :param latent_vector0: First latent vector.
    :param latent_vector1: Second latent vector.
    :param num_interpolation_steps: Number of interpolation steps.
    :param pos: Positional encodings.
    :param vae: Variational Autoencoder (VAE) model.
    :param ref_cp: Reference checkpoint for the model.
    :param inr: Implicit Neural Representation (INR) model.
    :return: List of interpolated images.
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
    """
    Generates a specified number of images and their reconstructions using a VAE, 
    and computes Mean Squared Error (MSE) between original and reconstructed images 
    and their weights.

    Args:
        vae: The variational autoencoder model to use for generating reconstructions.
        dataset: The dataset containing input samples and positions.
        inr: Implicit Neural Representation (INR) model used for rendering images.
        n_samples: The number of samples to process.
        random: Whether to select samples randomly or sequentially.

    Returns:
        images: Original images generated from the dataset.
        recons: Reconstructed images generated by the VAE.
        mse_weights: MSE values between original and reconstructed weights.
        mse_images: MSE values between original and reconstructed images.
    """
    # Select random or sequential indices for the samples
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

    # Get the positional encodings for the selected samples
    positions = dataset[0][2].unsqueeze(0).repeat(n_samples, 1, 1)

    ref_cp = dataset.get_state_dict(index=0)

    inputs = torch.stack([dataset[i][0] for i in idx])

    # Generate outputs using the VAE in evaluation mode
    vae.eval()
    with torch.no_grad():
        outputs,_,_,_ = vae(inputs, positions)

    # Convert tokenized weights back to state dictionaries for inputs and outputs    
    state_dicts_output = [tokens_to_weights(t, p, ref_cp) for t, p in zip(outputs, positions)]
    state_dicts_input = [tokens_to_weights(t, p, ref_cp) for t, p in zip(inputs, positions)]

    images = []
    recons = []
    mse_weights = []
    mse_images = []
    for i in range(n_samples):
        # Extract input and output state dictionaries for each sample
        state_dict_input = state_dicts_input[i]
        state_dict_output = state_dicts_output[i]

        # Render the original and reconstructed images
        image = compute_image(inr, state_dict_input)
        image_recon = compute_image(inr, state_dict_output)

        images.append(image)
        recons.append(image_recon)

        # Compute MSE between original and reconstructed images
        mse_images.append(calculate_mse(image, image_recon))

        # Flatten the weights for MSE computation
        flattened_weights_orig = weights_to_flattened_weights(state_dict_input)
        flattened_weights_recon = weights_to_flattened_weights(state_dict_output)

        # Compute MSE between original and reconstructed weights
        mse_weights.append(calculate_mse(flattened_weights_orig, flattened_weights_recon))
    return images, recons, mse_weights, mse_images


def generate_diffusion_images(diffusion_model, inr, n_samples, ref_cp=None):
    """
    Generates images using a diffusion model with or without an autoencoder.

    Args:
        diffusion_model: The diffusion model used to generate samples.
        inr: Implicit Neural Representation (INR) model for rendering images.
        n_samples: The number of images to generate.
        ref_cp: Reference state dictionary for reconstructing weights (optional).

    Returns:
        images: Generated images.
    """
    diffusion_model.eval()
    if diffusion_model.autoencoder is None:
        # Generate samples using hyperdiffusion
        samples = diffusion_model.generate_samples(n_samples)
        images = [compute_image(inr, flattened_weights_to_weights(s, inr)) for s in samples]
    else:
        # Generate samples using stable hyperdiffusion
        _, samples_reconstructed, positions = diffusion_model.generate_samples(n_samples)
        images = [compute_image(inr, tokens_to_weights(t, p, ref_cp)) for t, p in zip(samples_reconstructed, positions)]
    return images


def generate_nearest_neighbors(diffusion_model, inr, dataset, num_samples, use_latent=False, metric="cosine", k=1):
    """
    Finds the nearest neighbors for samples generated by the diffusion model in the dataset
    and computes MSE between generated samples and their nearest neighbors.

    Args:
        diffusion_model: The diffusion model used to generate samples.
        vae: Variational autoencoder model (for latent space comparison).
        inr: Implicit Neural Representation (INR) model for rendering images.
        dataset: The dataset containing samples and positions.
        num_samples: The number of samples to generate and compare.
        use_latent: Whether to compare in latent space or original space.
        metric: The distance metric to use for finding neighbors (e.g., "cosine").
        k: The number of nearest neighbors to find.

    Returns:
        dataset_images: Nearest neighbor images from the dataset.
        generated_images: Images generated by the diffusion model.
        mse_images: MSE values between generated images and nearest neighbors.
        mse_weights: MSE values between generated and nearest neighbor weights.
        distances: Distances between generated samples and their nearest neighbors.
    """
    if diffusion_model.autoencoder is None:
        # Generate samples using hyperdiffusion
        diffusion_outputs = diffusion_model.generate_samples(num_samples)
        dataset_samples = [dataset[i] for i in range(len(dataset))]
        dataset_samples = torch.stack(dataset_samples)

    else:
        # Generate samples using stable hyperdiffusion
        diffusion_latent_outputs, diffusion_outputs, diffusion_positions = diffusion_model.generate_samples(num_samples)
        dataset_samples = [dataset[i][0] for i in range(len(dataset))]
        dataset_positions = [dataset[i][2] for i in range(len(dataset))]
        dataset_samples = torch.stack(dataset_samples)
        dataset_positions = torch.stack(dataset_positions)

    # Store original outputs and dataset samples for reconstruction
    diffusion_org_outputs = diffusion_outputs
    dataset_samples_org = dataset_samples

    if use_latent and diffusion_model.autoencoder:
        diffusion_model.autoencoder.eval()
        with torch.no_grad():
            dataset_samples, _, _ = diffusion_model.autoencoder.encoder(dataset_samples, dataset_positions)
        diffusion_outputs = diffusion_latent_outputs

    # Flatten outputs for nearest neighbor search
    diffusion_outputs_flattened = diffusion_outputs.view(diffusion_outputs.size(0), -1)
    dataset_samples_flattened = dataset_samples.view(dataset_samples.size(0), -1)

    indices, distances = find_nearest_neighbor(diffusion_outputs_flattened, dataset_samples_flattened, metric=metric, k=k)
    nearest_samples = dataset_samples_org[indices]

    ref_cp = dataset.get_state_dict(index=0)

    if use_latent:
        diffusion_outputs = diffusion_org_outputs

    if diffusion_model.autoencoder is None:
        # Render nearest neighbor and generated images for hyperdiffusion
        dataset_images = [
            [
                compute_image(inr, flattened_weights_to_weights(s, inr))
                for s in nearest_samples[i]
            ]
            for i in range(nearest_samples.shape[0])
        ]
        generated_images = [compute_image(inr, flattened_weights_to_weights(s, inr)) for s in diffusion_outputs]
    else:
        # Render nearest neighbor and generated images for stable hyperdiffusion
        nearest_positions = dataset_positions[indices]
        dataset_images = [
            [
                compute_image(inr, tokens_to_weights(t, p, ref_cp))
                for t, p in zip(nearest_samples[i], nearest_positions[i])
            ]
            for i in range(nearest_samples.shape[0])
        ]
        generated_images = [
            compute_image(inr, tokens_to_weights(t, p, ref_cp)) for t, p in zip(diffusion_outputs, diffusion_positions)
        ]

    # Calculate MSE between generated images and nearest dataset images
    mse_images = [
        [calculate_mse(generated_images[i], dataset_images[i][j])  # Compare to each of the k nearest images
        for j in range(k)]
        for i in range(len(generated_images))
    ]

    # Calculate MSE between weights
    mse_weights = [
        [calculate_mse(diffusion_outputs[i], nearest_samples[i][j])
        for j in range(k)]
        for i in range(len(diffusion_outputs))
    ]

    return dataset_images, generated_images, mse_images, mse_weights, distances
import numpy as np
import torch
import einops
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM


def calculate_mse(original, reconstructed):
    """
    Calculate Mean Squared Error (MSE) between two images.

    :param original: Original image (numpy array or torch tensor).
    :param reconstructed: Reconstructed image (numpy array or torch tensor).
    :return: MSE value.
    """
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original)
    if isinstance(reconstructed, np.ndarray):
        reconstructed = torch.from_numpy(reconstructed)
    return torch.mean((original - reconstructed) ** 2).item()


def calculate_fid(real_images, generated_images):
    """
    Calculate the Frechet Inception Distance (FID) between real and generated images.
    :param real_images: Real images.
    :param generated_images: Generated images.
    :return: FID value.
    """
    fid = FrechetInceptionDistance(input_img_size=(3, 28, 28))

    real_images = einops.repeat(real_images, "b h w -> b c h w", c=3).to(torch.uint8)
    generated_images = einops.repeat(generated_images, "b h w -> b c h w", c=3).to(torch.uint8)

    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    fid_score = fid.compute()
    return fid_score


def find_nearest_neighbor(generated_samples, dataset_samples, metric="cosine",  k=1):
    if metric == "cosine":
        diffusion_outputs_norm = F.normalize(generated_samples, p=2, dim=1)  # Shape: (7000, 65*33)
        dataset_samples_norm = F.normalize(dataset_samples, p=2, dim=1)  # Shape: (num_samples, 65*33)
        similarities = torch.matmul(diffusion_outputs_norm, dataset_samples_norm.T)
    elif metric == "euclidean":
        similarities = torch.sqrt(torch.sum((generated_samples.unsqueeze(2) - dataset_samples.T.unsqueeze(0)) ** 2, dim=2))
    else:
        raise ValueError("Invalid metric. Choose 'cosine' or 'euclidean'.")
    
    #indices = torch.argmax(similarities, dim=1)  # Shape: (num_samples,)
    distances, indices = torch.topk(similarities, k=k, dim=1, largest=True)

    return indices, distances


def compute_mmd(X, Y, kernel="rbf", gamma=1.0):
    """
    Compute the Minimum Matching Distance (MMD) between two distributions.
    
    Args:
        X (np.ndarray): Generated samples (n_samples, n_features).
        Y (np.ndarray): Real samples (n_samples, n_features).
        kernel (str): Kernel type ('rbf' or 'linear').
        gamma (float): Bandwidth for the RBF kernel.

    Returns:
        float: MMD distance.
    """
    if kernel == "rbf":
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == "linear":
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")
    
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return np.sqrt(mmd)


def compute_coverage(X, Y, threshold=12, metric="euclidean"):
    """
    Compute Coverage (COV) between generated and real samples.
    
    Args:
        X (np.ndarray): Generated samples (n_samples, n_features).
        Y (np.ndarray): Real samples (n_samples, n_features).
        threshold (float): Distance threshold.
        metric (str): Distance metric (e.g., 'euclidean', 'cosine').

    Returns:
        float: Coverage metric (0 to 1).
    """
    distances = cdist(Y, X, metric=metric)  # Compute pairwise distances
    covered = (distances.min(axis=1) <= threshold).sum()
    return covered / len(Y)


def compute_1nna(X, Y, metric="euclidean"):
    """
    Compute 1-Nearest-Neighbor Accuracy (1-NNA) between generated and real samples.
    
    Args:
        X (np.ndarray): Generated samples (n_samples, n_features).
        Y (np.ndarray): Real samples (n_samples, n_features).
        metric (str): Distance metric (e.g., 'euclidean', 'cosine').

    Returns:
        float: 1-NNA accuracy (0 to 1, ideal is 0.5).
    """
    data = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))
    
    # Use a 1-nearest neighbor classifier with leave-one-out strategy
    neigh = NearestNeighbors(n_neighbors=2, metric=metric)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    
    # Exclude self-match (first neighbor is always the sample itself)
    correct = 0
    for i, (dist, idx) in enumerate(zip(distances[:, 1], indices[:, 1])):
        if labels[i] == labels[idx]:
            correct += 1
    
    accuracy = correct / len(data)
    return accuracy



def detect_novelty(features_train, features_generated, threshold=10.0):
    distances = euclidean_distances(features_generated, features_train)
    min_distances = distances.min(axis=1)  # Minimum distance to any training sample
    novelty_mask = min_distances > threshold
    return novelty_mask, min_distances

#def detect_low_likelihood_samples(model, samples, threshold=-50):
#    log_likelihoods = model.compute_log_likelihood(samples)  # Replace with your method
#    return log_likelihoods < threshold


def novelty_svm(features_train, features_generated):
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    svm.fit(features_train)
    predictions = svm.predict(features_generated)  # -1 indicates novelty
    return predictions

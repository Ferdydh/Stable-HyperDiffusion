import torch
import random
from typing import Tuple, Optional, List
from torchvision.transforms import RandomErasing
import einops
from src.core.config import BaseExperimentConfig


def identity_transform(tensor, mask, pos):
    """Identity transformation that returns two identical views."""
    return tensor, mask, pos, tensor, mask, pos


def add_noise(
    tensor: torch.Tensor, sigma: float = 0.1, multiplicative: bool = True
) -> torch.Tensor:
    """Add Gaussian noise to tensor."""
    noise = torch.randn_like(tensor) * sigma
    if multiplicative:
        return tensor * (1.0 + noise)
    return tensor + noise


def apply_erasing(
    tensor: torch.Tensor, p: float = 0.5, value: float = 0
) -> torch.Tensor:
    """Apply random erasing to tensor."""
    erasing = RandomErasing(
        p=p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=value, inplace=False
    )
    # Reshape for RandomErasing and back
    tensor = einops.rearrange(
        tensor, "b n d -> (b n) 1 d"
    )  # b=batch, n=num_tokens, d=token_dim
    tensor = erasing(tensor)
    return einops.rearrange(tensor, "(b n) 1 d -> b n d")


def cut_window(
    tensor: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut a random window from the sequence."""
    seq_len = tensor.shape[1]
    start_idx = (
        0 if seq_len == window_size else random.randint(0, seq_len - window_size)
    )
    end_idx = start_idx + window_size

    # Use torch indexing to select window
    return (
        tensor[:, start_idx:end_idx, :],  # [batch, num_tokens, token_dim]
        mask[:, start_idx:end_idx, :],  # [batch, num_tokens, token_dim]
        pos[:, start_idx:end_idx, :],  # [batch, num_tokens, token_dim]
    )


def stack_batches(
    tensor: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack batch dimension into sequence dimension."""
    return (
        einops.rearrange(
            tensor, "b n d -> (b n) 1 d"
        ),  # b=batch, n=num_tokens, d=token_dim
        einops.rearrange(mask, "b n d -> (b n) 1 d"),
        einops.rearrange(pos, "b n d -> (b n) 1 d"),
    )


def select_permutation(
    tensor: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, canonical: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select either canonical (first) or random permutation of sequence."""
    if canonical:
        return tensor, mask, pos

    # Generate random permutation indices for each batch
    batch_size, seq_len, _ = tensor.shape
    perm_idx = torch.stack(
        [torch.randperm(seq_len, device=tensor.device) for _ in range(batch_size)]
    )

    # Create batch indices for advanced indexing
    batch_idx = torch.arange(batch_size, device=tensor.device)[:, None]

    # Use torch advanced indexing for permutation
    return (
        tensor[batch_idx, perm_idx],  # [batch, num_tokens, token_dim]
        mask[batch_idx, perm_idx],  # [batch, num_tokens, token_dim]
        pos[batch_idx, perm_idx],  # [batch, num_tokens, token_dim]
    )


def create_transform(
    window_size: int,
    multi_windows: bool,
    noise_view1: float,
    noise_view2: float,
    erase_view1: Optional[float],
    erase_view2: Optional[float],
    use_permutation: bool,
    view1_canonical: bool,
    view2_canonical: bool,
) -> callable:
    """Create a transform function with the specified parameters."""

    def transform(
        tensor: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Apply window cutting or batch stacking
        # if multi_windows:
        #     tensor, mask, pos = stack_batches(tensor, mask, pos)
        # else:
        #     tensor, mask, pos = cut_window(tensor, mask, pos, window_size)

        # Create two views through permutation or copying
        if use_permutation:
            tensor1, mask1, pos1 = select_permutation(
                tensor, mask, pos, view1_canonical
            )
            tensor2, mask2, pos2 = select_permutation(
                tensor, mask, pos, view2_canonical
            )
        else:
            tensor1, mask1, pos1 = tensor.clone(), mask.clone(), pos.clone()
            tensor2, mask2, pos2 = tensor.clone(), mask.clone(), pos.clone()

        # Apply noise augmentation
        if noise_view1 > 0:
            tensor1 = add_noise(tensor1, noise_view1)
        if noise_view2 > 0:
            tensor2 = add_noise(tensor2, noise_view2)

        # Apply erasing augmentation
        if erase_view1 is not None:
            tensor1 = apply_erasing(tensor1, erase_view1)
        if erase_view2 is not None:
            tensor2 = apply_erasing(tensor2, erase_view2)

        return tensor1, mask1, pos1, tensor2, mask2, pos2

    return transform


def setup_transformations(config: BaseExperimentConfig):
    """Setup training, validation and downstream transforms based on config."""
    if not config.augmentations.apply_augmentations:
        return identity_transform, identity_transform, identity_transform

    # Training transforms
    train_transform = create_transform(
        window_size=config.model.window_size,
        multi_windows=config.augmentations.multi_windows_train,
        noise_view1=config.augmentations.add_noise_view_1_train,
        noise_view2=config.augmentations.add_noise_view_2_train,
        erase_view1=config.augmentations.erase_augment_view_1_train,
        erase_view2=config.augmentations.erase_augment_view_2_train,
        use_permutation=config.augmentations.permutation_number_train > 0,
        view1_canonical=config.augmentations.view_1_canon_train,
        view2_canonical=config.augmentations.view_2_canon_train,
    )

    # Validation transforms
    val_transform = create_transform(
        window_size=config.model.window_size,
        multi_windows=config.augmentations.multi_windows_train,
        noise_view1=config.augmentations.add_noise_view_1_val,
        noise_view2=config.augmentations.add_noise_view_2_val,
        erase_view1=config.augmentations.erase_augment_view_1_val,
        erase_view2=config.augmentations.erase_augment_view_2_val,
        use_permutation=config.augmentations.permutation_number_val > 0,
        view1_canonical=config.augmentations.view_1_canon_val,
        view2_canonical=config.augmentations.view_2_canon_val,
    )

    # Downstream transform
    def dst_transform(tensor: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor):
        """Simple transform that optionally selects canonical sequence."""
        if config.augmentations.permutation_number_train > 0:
            return select_permutation(tensor, mask, pos, canonical=True)
        return tensor, mask, pos

    return train_transform, val_transform, dst_transform

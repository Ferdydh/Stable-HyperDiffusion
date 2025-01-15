import copy
import logging
import torch

from src.data.inr import INR


def weights_to_flattened_weights(state_dict: dict) -> torch.Tensor:
    """Flatten state_dict to weights."""

    weights = []
    for weight in state_dict.values():
        weights.append(weight.flatten())
    return torch.hstack(weights)


def flattened_weights_to_weights(weights: torch.Tensor, inr_model: INR) -> INR:
    """Load flattened weights into INR model."""
    state_dict = {}
    start_idx = 0
    for key, param in inr_model.state_dict().items():
        param_size = param.numel()
        param_data = weights[start_idx : start_idx + param_size].reshape(param.shape)
        state_dict[key] = param_data
        start_idx += param_size
    return state_dict


def weights_to_tokens(
    checkpoint, tokensize: int, return_mask: bool = True, ignore_bn=False, device=None
):
    """
    transforms a checkpoint into a sequence of tokens, one token per channel / neuron
    Tokensize can be set to 0 to automatically discover the correct size (maximum) size
    if tokensize is smaller than the maximum size, the tokens will be chunked into tokens of size tokensize
    tokens are zero-padded to tokensize
    masks indicate with 1 where the original tokens were, and 0 where the padding is
    Args:
        checkpoint: checkpoint to be vectorized
        tokensize: int output dimension of each token
        return_mask: bool wether to return the mask of nonzero values
    Returns
        tokens: list of tokens or zero padded tensor of tokens
        mask: mask of nonzero values
        pos: tensor with 3d positions for every token in the vectorized model sequence
    """
    # init output
    tokens = []
    pos = []
    masks = []

    #### Discover Tokensize ####################################################
    if tokensize == 0:
        # discover tokensize
        tokensize = 0
        for key in checkpoint.keys():
            # get valid layers
            # check for batchnorm layers
            if "bn" in key or "downsample.1" in key or "batchnorm" in key:
                # ignore all batchnorm layers if ignore_bn is set
                if ignore_bn:
                    continue
                # otherwise check for other keys in all remaining layers
            # get weights of all layers
            if "weight" in key:
                tmp = checkpoint[key].shape
            # get running mean and var of batchnorm layers
            elif "running_mean" in key or "running_var" in key:
                tmp = checkpoint[key].shape
            else:
                continue
            tempsize = torch.prod(torch.tensor(tmp)) / tmp[0]  # noqa: F821
            # cat biases to channels if they exist in checkpoint
            if key.replace("weight", "bias") in checkpoint:
                tempsize += 1

                if tempsize > tokensize:
                    tokensize = tempsize
        # for key in checkpoint.keys():
        #     if "weight" in key:
        #         # get correct slice of modules out of vec sequence
        #         if ignore_bn and ("bn" in key or "downsample.1" in key):
        #             continue
        #         tmp = checkpoint[key].shape
        #         tempsize = torch.prod(torch.tensor(tmp)) / tmp[0]
        #         # cat biases to channels if they exist in checkpoint
        #         if key.replace("weight", "bias") in checkpoint:
        #             tempsize += 1

        #         if tempsize > tokensize:
        #             tokensize = tempsize

    # get raw tokens and positions
    tokensize = int(tokensize)

    #### Get Tokens ####################################################
    idx = 0
    # use only weights and biases
    for key in checkpoint.keys():
        # if "weight" in key:
        #     #### get weights ####
        #     if ignore_bn and ("bn" in key or "downsample.1" in key):
        #         continue
        # get valid layers
        # check for batchnorm layers
        if "bn" in key or "downsample.1" in key or "batchnorm" in key:
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key or "running_mean" in key or "running_var" in key:
            w = checkpoint[key]
            # flatten to out_channels x n
            w = w.view(w.shape[0], -1)
            # cat biases to channels if they exist in checkpoint
            if "weight" in key:
                if key.replace("weight", "bias") in checkpoint:
                    b = checkpoint[key.replace("weight", "bias")]
                    w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)

            #### infer positions ####
            # infer # of tokens per channel
            a = w.shape[1] // tokensize
            b = w.shape[1] % tokensize
            token_factor = int(a)
            if b > 0:
                token_factor += 1

            # get positions, repeating for parts of the same token (overall position will be different)
            idx_layer = [
                [idx, jdx] for jdx in range(w.shape[0]) for _ in range(token_factor)
            ]
            # increase layer counter
            idx += 1
            # add to overall position
            pos.extend(idx_layer)

            #### tokenize ####
            # if b> 0, weights need to be zero-padded
            if b > 0:
                # start with the mask (1 where there is a weight, 0 for padding)
                mask = torch.zeros(w.shape[0], tokensize * token_factor)
                mask[:, : w.shape[1]] = torch.ones(w.shape)
                # zero pad the end of w in dim=1 so that shape[1] is multiple of tokensize
                w_tmp = torch.zeros(w.shape[0], tokensize * token_factor)
                w_tmp[:, : w.shape[1]] = w
                w = w_tmp
            else:
                mask = torch.ones(w.shape[0], tokensize * token_factor)

            # break along token-dimension
            w = w.view(-1, tokensize)
            mask = mask.view(-1, tokensize).to(torch.bool)

            # extend out with new tokens, zero's (and only entry) is a list
            #if device:
            #    w = w.to(device)
            #    mask = mask.to(device)
            tokens.append(w)
            masks.append(mask)

    #### postprocessing ####################################################
    # cat tokens / masks
    tokens = torch.cat(tokens, dim=0)
    masks = torch.cat(masks, dim=0)

    # add index tensor over whole sequence
    pos = [(ndx, idx, jdx) for ndx, (idx, jdx) in enumerate(pos)]
    pos = torch.tensor(pos, dtype=torch.int32)
    # cast tensor to int16
    # if pos.max() > 32767:
    #     logging.debug(
    #         f"max position value of {pos.max()} does not fit into torch.int16 range. Change data type"
    #     )
    #     pos = pos.to(torch.int)
    # else:
    #     pos = pos.to(torch.int16)
    #if device:
    #    pos.to(device)
    if return_mask:
        return tokens, masks, pos
    else:
        return tokens, pos


def tokens_to_weights(tokens, pos, reference_checkpoint, ignore_bn=False):
    """
    casts sequence of tokens back to checkpoint
    Args:
        tokens: sequence of tokens
        pos: sequence of positions
        reference_checkpoint: reference checkpoint to be used for shape information
        ignore_bn: bool wether to ignore batchnorm layers
    Returns
        checkpoint: checkpoint with weights and biases
    """
    # make copy to prevent memory management issues
    checkpoint = copy.deepcopy(reference_checkpoint)
    # use only weights and biases
    idx = 0
    for key in checkpoint.keys():
        # if "weight" in key:
        #     # get correct slice of modules out of vec sequence
        #     if ignore_bn and ("bn" in key or "downsample.1" in key):
        #         continue
        if "bn" in key or "downsample.1" in key or "batchnorm" in key:
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key or "running_mean" in key or "running_var" in key:
            # get modules shape
            mod_shape = checkpoint[key].shape

            # get slice for current layer
            idx_channel = torch.where(pos[:, 1] == idx)[0]
            w_t = torch.index_select(input=tokens, index=idx_channel, dim=0)

            # infer length of content
            contentlength = int(torch.prod(torch.tensor(mod_shape)) / mod_shape[0])

            # update weights
            checkpoint[key] = w_t.view(mod_shape[0], -1)[:, :contentlength].view(
                mod_shape
            )

            # check for bias
            if "weight" in key:
                if key.replace("weight", "bias") in checkpoint:
                    checkpoint[key.replace("weight", "bias")] = w_t.view(
                        mod_shape[0], -1
                    )[:, contentlength]

            # if key.replace("weight", "bias") in checkpoint:
            # checkpoint[key.replace("weight", "bias")] = w_t.view(mod_shape[0], -1)[
            #     :, contentlength
            # ]

            # update counter
            idx += 1

    return checkpoint

import torch
from torchvision.transforms import RandomErasing
import random


#############################################################################
class AugmentationPipeline(torch.nn.Module):
    """
    Wrapper around a stack of augmentation modules
    Handles passing data through stack, isolating properties
    """

    def __init__(self, stack, keep_properties: bool = False):
        """
        passes stream of data through stack
        """
        super(AugmentationPipeline, self).__init__()
        self.stack = stack
        if keep_properties:
            self.forward = self._forward_props
        else:
            self.forward = self._forward

    def _forward(self, ddx, mdx, p, props=None):
        # apply stack 1
        out = (ddx, mdx, p)
        for m in self.stack:
            out = m(*out)
        return out

    def _forward_props(self, ddx, mdx, p, props):
        # apply stack 1
        out = (ddx, mdx, p, props)
        for m in self.stack:
            out = m(*out)
        return out


#############################################################################
class TwoViewSplit(torch.nn.Module):
    """ """

    def __init__(
        self,
        stack_1,
        stack_2,
        mode: str = "copy",
        view_1_canon: bool = True,
        view_2_canon: bool = False,
    ):
        """
        splits input stream of ddx, mask, p in two streams
        passes two streams through stack_1, stack_2, respectively
        if mode == "copy", then ddx, mask, p are cloned
        if mode == "permutation", then mask, p are copied, ddx is sliced along first axis to get permuted versions
        """
        super(TwoViewSplit, self).__init__()
        self.stack_1 = stack_1
        self.stack_2 = stack_2

        self.mode = mode
        if self.mode == "copy":
            self.forward = self._forward_copy
        elif self.mode == "permutation":
            self.forward = self._forward_permutation
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

        self.view_1_canon = view_1_canon
        self.view_2_canon = view_2_canon

    def _forward_copy(self, ddx1, mdx1, p1):
        # clone ddx, mdx
        ddx2, mdx2, p2 = (
            ddx1.clone().to(ddx1.device),
            mdx1.clone().to(ddx1.device),
            p1.clone().to(ddx1.device),
        )
        # apply stack 1
        for m in self.stack_1:
            ddx1, mdx1, p1 = m(ddx1, mdx1, p1)
        # apply stack 2
        for m in self.stack_2:
            ddx2, mdx2, p2 = m(ddx2, mdx2, p2)
        return ddx1, mdx1, p1, ddx2, mdx2, p2

    def _forward_permutation(self, ddx1, mdx1, p1):
        # ddx.shape[-3] contains random permutations
        # choose two out of those and slice
        perm_ids = torch.randperm(
            n=ddx1.shape[-3], dtype=torch.int32, device=ddx1.device
        )[:2]
        if self.view_1_canon == True:
            perm_ids[0] = 0
        if self.view_2_canon == True:
            perm_ids[1] = 0
        # slice+clone second sample first
        # logging.debug(f"perm_ids: {perm_ids}")
        ddx2 = (
            torch.index_select(ddx1.clone(), -3, perm_ids[1]).squeeze().to(ddx1.device)
        )
        # slice / overwrite first sample
        ddx1 = torch.index_select(ddx1, -3, perm_ids[0]).squeeze().to(ddx1.device)

        # clone mdx, p
        mdx2, p2 = mdx1.clone().to(ddx1.device), p1.clone().to(ddx1.device)

        # apply stack 1
        for m in self.stack_1:
            ddx1, mdx1, p1 = m(ddx1, mdx1, p1)
        # apply stack 2
        for m in self.stack_2:
            ddx2, mdx2, p2 = m(ddx2, mdx2, p2)
        return ddx1, mdx1, p1, ddx2, mdx2, p2


#############################################################################
class PermutationSelector(torch.nn.Module):
    """
    ffcv batches use the first dimension to store random permutations of the same sample
    at inference, these need to be separatered and a single version picked.
    this module does that
    """

    def __init__(
        self,
        mode: str = "identity",
        keep_properties: bool = False,
    ):
        """
        if mode == "random", then random permutation is chosen
        if mode == "canonical", ddx[0] is chosen
        """
        super(PermutationSelector, self).__init__()
        self.keep_properties = keep_properties
        self.mode = mode
        if self.mode == "random":
            self.forward = self._forward_random
        elif self.mode == "canonical":
            self.forward = self._forward_canonical
        elif self.mode == "identity":
            self.forward = self._forward_identity
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

    def _return(self, ddx, mdx, p, props):
        if self.keep_properties:
            return ddx, mdx, p, props
        else:
            return ddx, mdx, p

    def _forward_canonical(self, ddx, mdx, p, props=None):
        # ddx.shape[0] contains random permutations
        # choose first out of those and slice
        # ddx = ddx[0]
        ddx = (
            torch.index_select(ddx, -3, torch.tensor(0).to(torch.int32))
            .squeeze()
            .to(ddx.device)
        )
        return self._return(ddx, mdx, p, props)

    def _forward_random(self, ddx, mdx, p, props=None):
        # ddx.shape[0] contains random permutations
        # choose one out of those and slice
        # -3 is the index of permutations
        perm_ids = torch.randperm(
            n=ddx.shape[-3], dtype=torch.int32, device=ddx.device
        )[:1]
        # logging.debug(f"perm_ids: {perm_ids}")
        # ddx = ddx[perm_ids[0]]
        ddx = torch.index_select(ddx, -3, perm_ids[0]).squeeze().to(ddx.device)

        return self._return(ddx, mdx, p, props)

    def _forward_identity(self, ddx, mdx, p, props=None):
        # pass through data without change
        return self._return(ddx, mdx, p, props)


#############################################################################
class WindowCutter(torch.nn.Module):
    """
    cuts random window chunks out of sequence of tokens
    Args:
        windowsize: size of window
    Returns:
        ddx: torch.tensor sequence of weight/channel tokens
        mdx: torch.tensor sequence of mask tokens
        p: torch.tensor sequence of positions
    """

    def __init__(self, windowsize: int = 12, keep_properties: bool = False):
        super(WindowCutter, self).__init__()
        self.windowsize = windowsize
        self.keep_properties = keep_properties

    def forward(self, ddx, mdx, p, props=None):
        """
        #TODO
        """
        # get lenght of token sequence
        max_len = ddx.shape[-2]
        # sample start
        if max_len == self.windowsize:
            idx_start = 0
        else:
            idx_start = random.randint(0, max_len - self.windowsize)
        idx_end = idx_start + self.windowsize

        # get index tensor
        idx = torch.arange(start=idx_start, end=idx_end, device=ddx.device)

        # apply window
        ddx = torch.index_select(ddx, -2, idx)
        mdx = torch.index_select(mdx, -2, idx)
        p = torch.index_select(p, -2, idx)

        if self.keep_properties:
            return ddx, mdx, p, props
        return ddx, mdx, p


#############################################################################
class MultiWindowCutter(torch.nn.Module):
    """
    cuts k random window chunks out of one sample of sequence of tokens
    Args:
        windowsize: size of window
        k: number of windows. k=1 is equivalent to WindowCutter. Rule of thumb can be: lenght of sequence / windowsize to get full coverage of sample
    Returns:
        ddx: torch.tensor sequence of weight/channel tokens
        mdx: torch.tensor sequence of mask tokens
        p: torch.tensor sequence of positions
    """

    def __init__(self, windowsize: int = 12, k: int = 10):
        super(MultiWindowCutter, self).__init__()
        self.windowsize = windowsize
        self.k = k

    def forward(self, ddx, mdx, p):
        # get lenght of token sequence

        # single sample case: match batch dimension
        if len(ddx.shape) == 2:
            ddx = ddx.unsqueeze(dim=0)
            mdx = mdx.unsqueeze(dim=0)
            p = p.unsqueeze(dim=0)

        # get max index
        max_idx = ddx.shape[1] - self.windowsize + 1

        # draw k random start indices
        idx_starts = torch.randint(0, max_idx, (self.k,))

        # apply slicing
        ddx = [
            ddx[:, idx_start : idx_start + self.windowsize, :]
            for idx_start in idx_starts
        ]
        mdx = [
            mdx[:, idx_start : idx_start + self.windowsize, :]
            for idx_start in idx_starts
        ]
        p = [
            p[:, idx_start : idx_start + self.windowsize, :] for idx_start in idx_starts
        ]

        # cat along batch dimension
        ddx = torch.cat(ddx, dim=0)
        mdx = torch.cat(mdx, dim=0)
        p = torch.cat(p, dim=0)

        # return
        return ddx, mdx, p


class StackBatches(torch.nn.Module):
    """
    stack batches from multi-window cutter to regular batches
    """

    def __init__(
        self,
    ):
        super(StackBatches, self).__init__()

    def forward(self, ddx, mdx, p):
        # stack along first two dimensions
        ddx = ddx.view((ddx.shape[0] * ddx.shape[1], ddx.shape[2], ddx.shape[3]))
        mdx = mdx.view((mdx.shape[0] * mdx.shape[1], mdx.shape[2], mdx.shape[3]))
        p = p.view((p.shape[0] * p.shape[1], p.shape[2], p.shape[3]))
        return ddx, mdx, p


#############################################################################
class ErasingAugmentation(torch.nn.Module):
    """
    #TODO
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: tuple = (0.02, 0.33),
        ratio: tuple = (0.3, 3.3),
        value=0,
    ):
        super(ErasingAugmentation, self).__init__()
        self.re = RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=True
        )

    def forward(self, ddx, mdx, p):
        """
        #TODO
        """
        # unsquezee along channel dimension to match torch random erasing logic
        ddx = ddx.unsqueeze(dim=-3)
        # apply inplace erasing
        self.re(ddx)
        # squeeze back again
        ddx = ddx.squeeze()
        return ddx, mdx, p


#############################################################################
class NoiseAugmentation(torch.nn.Module):
    """ """

    def __init__(self, sigma: float = 0.1, multiplicative_noise: bool = True):
        super(NoiseAugmentation, self).__init__()
        self.sigma = sigma
        if multiplicative_noise:
            self.forward = self._forward_multiplicative
        else:
            self.forward = self._forward_additive

    def _forward_multiplicative(self, ddx, mdx, p):
        ddx = ddx * (1.0 + self.sigma * torch.randn(ddx.shape, device=ddx.device))
        return ddx, mdx, p

    def _forward_additive(self, ddx, mdx, p):
        ddx = ddx + self.sigma * torch.randn(ddx.shape, device=ddx.device)
        return ddx, mdx, p


#############################################################################

import torch


def split_tensor(
        tensor,
        tensor_channels=3,
        chunks=2,
        dim=1,
        height=64,
        width=64
):
    """
    Split a tensor-block by block size (into per-image tensors)
    This is basically a wrapper for torch.chunk which reshapes the result tensors
    :param tensor: tensor to be split
    :param tensor_channels: number of channels C
    :param chunks: number of chunks to split the tensor into
    :param dim: along which dim to split, must correspond to time axis, i.e. 1 for (N, T, C, H, W)
    :return: list of tensor parts
    """
    parts = torch.chunk(tensor, chunks=chunks, dim=dim)
    return [part.view(-1, tensor_channels, height, width) for part in parts]

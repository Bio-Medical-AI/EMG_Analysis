"""
Transforms meant to be used by dataloaders to modify data that is going to be passed to model.
"""
import os
import numpy as np
import torch
from scipy.ndimage import median_filter
from torchvision import transforms


def spatial_median_filter(frame: np.ndarray) -> np.ndarray:
    """
    Calculate a multidimensional median filter.
    Args:
        frame: 2d numpy array

    Returns:
        Filtered array. Has the same shape as input.

    """
    return median_filter(frame, size=(3, 3))


def load_ndarray(path: str or os.PathLike) -> np.ndarray:
    """
    Load an array from file.
    Args:
        path: The file to read. File-like objects must support the seek() and read() methods and must always be opened in binary mode. Pickled files require that the file-like object support the readline() method as well.

    Returns:
        Data stored in the file. For .npz files, the returned instance of NpzFile class must be closed to avoid leaking file descriptors.

    """
    return np.load(path)


def ndarray_from_str(rep: str) -> np.ndarray:
    """
    A new 1-D array initialized from text data in a string.
    Args:
        rep: A string containing the data

    Returns:
        The constructed array.
    """
    return np.fromstring(rep)


def fix_ndarray_to_tensor_3d(img: torch.Tensor) -> torch.Tensor:
    """
    Performing permutation of dimensions on copy of given tensor swapping them like this: 0 -> 1, 1 -> 2, 2 -> 0.
    Args:
        img: 3D Tensor to permute.

    Returns:
        Copy of tensor with permuted dimensions.
    """
    return img.permute((1, 2, 0)).contiguous()


SPATIAL_MEDIAN_FILTER = transforms.Lambda(spatial_median_filter)
LOAD_NDARRAY = transforms.Lambda(load_ndarray)
NDARRAY_FROM_STR = transforms.Lambda(ndarray_from_str)
FIX_NDARRAY_TO_TENSOR_3D = transforms.Lambda(fix_ndarray_to_tensor_3d)

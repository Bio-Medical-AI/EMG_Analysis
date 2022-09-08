import numpy as np
from scipy.ndimage import median_filter
from torchvision import transforms


def spatial_median_filter(frame):
    return median_filter(frame, size=(3, 3))


def load_ndarray(path):
    return np.load(path)


def ndarray_from_str(rep):
    return np.fromstring(rep)


SPATIAL_MEDIAN_FILTER = transforms.Lambda(spatial_median_filter)
LOAD_NDARRAY = transforms.Lambda(load_ndarray)
NDARRAY_FROM_STR = transforms.Lambda(ndarray_from_str)

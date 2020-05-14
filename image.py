"""Module for processing small RGB images"""
from PIL import Image
from numpy import asarray


def image_to_array(file):
    """Read image file into Numpy array"""
    with Image.open(file) as img:
        data = asarray(img)
    return data


def reshape_image(data):
    """Reshape RGB data into 2D array"""
    return data.reshape(-1, 3)


def scale_image(data):
    """Scale image data"""
    return data / 255


def rescale_image(data):
    """Rescale image data"""
    return data * 255


def load_data_from_image(file):
    """Read and process image data into Numpy array"""
    return scale_image(reshape_image(image_to_array(file)))

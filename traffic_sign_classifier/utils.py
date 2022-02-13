import csv
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Optional

import numpy as np


def load_signnames(signnames_file) -> Dict[int, str]:
    """Return id to sign-name mapping as a dict"""
    id_to_sign_name_dict = {}
    with open(signnames_file, "r") as csv_file:
        id_to_sign_name_dict = {
            int(row["ClassId"]): row["SignName"] for row in csv.DictReader(csv_file)
        }
    return id_to_sign_name_dict


def group_by_category(
    labels: List[int], sample_size: Optional[int] = None, shuffle: bool = False
):
    """Return items/example indices grouped by category"""
    idx_by_category = defaultdict(lambda: [])
    for idx, category in enumerate(labels):
        idx_by_category[category].append(idx)
    if sample_size:
        return {
            category: [
                items[index]
                for index in np.random.permutation(len(items))[:sample_size]
            ]
            if shuffle
            else items[:sample_size]
            for category, items in idx_by_category.items()
        }
    return idx_by_category


def accuracy_classes(prediction: np.ndarray, gt: np.ndarray):
    """Calculate accuracy from classes/categories prediction array (1D)"""
    assert len(prediction) == len(gt)
    return float(np.sum(prediction == gt)) / len(gt)


def accuracy(prediction, gt):
    """Calculate accuracy from logits (or softmax output) predictions"""
    return accuracy_classes(np.argmax(prediction, axis=1), gt)


def inspect_prediction(model, input_images, ground_truth) -> np.ndarray:
    """Inspect prediction
    :param model: trained model
    :param input_image: images to predict as a numpy array `N x Weight x Height x channel`
    :param ground_truth: images label (one hot encoded to check !)
    :return :  prediction success as an array of boolean
    """
    return np.array(np.argmax(model.predict(input_images), axis=1)) == np.array(
        ground_truth
    )


def stack_pixels(pixel_channel_array, width=32, height=32):
    """Stack pixel channels array to build an array of 32x32 images"""
    return np.stack((np.stack((pixel_channel_array,) * 32, axis=1),) * 32, axis=1)


def channel_min_max_normalizer(image_dataset: np.ndarray):
    """(local) min-max normalizer (equalizer)
    :param image_dataset:  numpy array with shape [N, width, height, channels_nbr] where N is the size of the image dataset

    """
    image_min = stack_pixels(np.min(image_dataset, axis=(1, 2))).astype(float)
    image_max = stack_pixels(np.max(image_dataset, axis=(1, 2))).astype(float)
    return (image_dataset - image_min) / (image_max - image_min)


def normalize(images: np.ndarray):
    """Center and nomrlize the images data from [0, 255] to [-1,1[
    (expects image channels values to be in [0, 255]) interval"""
    return (images - 128.0) / 128.0

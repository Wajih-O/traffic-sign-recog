""" Image grid visualization tools
@author Wajih Ouertani
@email wajih.ouertani@gmail.com
"""
import logging

import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from traffic_sign_classifier.utils import group_by_category

LOGGER = logging.getLogger()


def show_image(ax, image: np.ndarray, title: Optional[str] = None, **kwargs):
    """a helper to show/plot image in a subplot (AxesSubplot)"""
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(image, **kwargs)
    if title:
        ax.set_title(f"{title}", pad=-2)


def gen_preview_output_file(
    label: Union[int, str], output_dir_path: str, output_ext: str = "jpg"
) -> str:
    """Generate output file for a group of categories"""
    output_file_path = ".".join(["_".join(["preview", str(label)]), output_ext])
    return os.path.join(output_dir_path, output_file_path)


def grid_visu(
    images: np.ndarray,
    labels: List[int],
    sample_size: int = 3,
    prediction_labels: Optional[List[int]] = None,
    categories: Optional[List[int]] = None,
    label_to_name: Optional[Dict[int, str]] = None,
    shuffle: bool = False,
    categories_per_fig: int = 5,  # number of category per figure
    output_dir_path: Optional[str] = None,
    output_ext: str = "jpg",
):
    """Visualize dataset/images sample (from the dataset) in a grid

    :param images: images dataset n*w*h*c. Where n is the data size
    :param labels: categories ground truth
    :param sample_size: sample size (to show) per class/category
    :param categories: category subset to consider in the visualization (default None -> all)
    :param label_to_name: a mapping dictionnary from class id to class name
    :param shuffle: shuffle data set when selecting the category sample
    :param categories_per_fig: the number of categories to show/display per-figure (to better load/render)
    """

    # TODO: ensure output_dir is a direcotry to contain all the preview if it is not none

    # Group the images by category/class
    by_category = group_by_category(labels, sample_size=sample_size, shuffle=shuffle)

    # Optionally filter the categories (to visualize if categories is set in param)
    if categories is not None:
        by_category = {
            key: by_category[key]
            for key in set(categories).intersection(by_category.keys())
        }

    sorted_categories = sorted(by_category.keys())  # sort selected categories
    if not label_to_name:
        label_to_name = {}

    # group categories in separate figures (loading faster)
    for index, grp in enumerate(
        [
            sorted_categories[i : i + categories_per_fig]
            for i in range(0, len(sorted_categories), categories_per_fig)
        ]
    ):
        subfigs = plt.figure(
            # constrained_layout=True,
            figsize=(2 * sample_size, 2 * len(grp)),
            tight_layout=False,
        ).subfigures(len(grp), 1, wspace=0.01)
        if not isinstance(subfigs, Iterable):
            subfigs = [subfigs]

        for category, fig in zip(grp, subfigs):
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, len(by_category[category])),
                axes_pad=0.1,
            )
            for ax, idx in zip(grid, by_category[category]):
                title = ""
                if idx is not None:
                    if prediction_labels:
                        title = prediction_labels[idx]
                    show_image(ax, images[idx], title)
            fig.suptitle(
                f"(class {category})  {label_to_name.get(category, category)}",
                fontsize="large",
            )
        # Saving the preview
        if output_dir_path is not None:
            # Ensure that output directory exists if not create it
            if not os.path.exists(output_dir_path):
                # Create the directory
                os.mkdir(output_dir_path)

            if os.path.isdir(output_dir_path):
                plt.savefig(
                    gen_preview_output_file(
                        index, output_dir_path, output_ext=output_ext
                    )
                )
            else:
                LOGGER.warning(f"{output_dir_path} exists and is not a directory")


def visualize_feature_map_output(
    images,
    paths: List[str],
    activation_model,
    layer_label,
    output_dir_path: Optional[str] = None,
    output_ext: str = "jpg",
    path_labeler=Callable[[str], int],
    class2label: Optional[Dict] = None,
):
    """Visualize the feature map output

    :param images: images np.array to be fed to the network (for which the feature map will be produced)
    :param paths: images paths
    :param activation_model: activation model
    :param layer_label: layer_name as the selector for the layer output/feature map to be visualized
    :param images_per_figure: the number of image to group by figure

    """
    # Sanity check
    assert len(paths) == len(images)

    # Building the group by paths/labeler
    grouped = defaultdict(lambda: [])
    for idx, label in zip(range(len(paths)), map(path_labeler, paths)):
        grouped[label].append(idx)

    activation_output = activation_model.predict(np.array(images)).transpose(0, 3, 1, 2)

    # Group activation map images by category
    feature_map_nbr = activation_output.shape[1]
    offset = 0
    for category, items in grouped.items():
        grp = activation_output[items]
        subfigs = plt.figure(
            # constrained_layout=True,
            figsize=(2 * grp.shape[1], 2 * len(grp)),
            tight_layout=False,
        ).subfigures(grp.shape[0], 1, wspace=0.01)

        if not isinstance(subfigs, Iterable):
            subfigs = [subfigs]

        for image_idx, ft_map, fig in zip(items, grp, subfigs):
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, feature_map_nbr),
                axes_pad=0.4,
            )
            for ax, ft_map_idx in zip(grid, range(ft_map.shape[0])):
                title = f"FeatureMap {ft_map_idx}"
                show_image(ax, ft_map[ft_map_idx], title, cmap="plasma")

            category_label = category
            if class2label is not None and category in class2label:
                category_label = f"(class-{category}) {class2label[category]}"
            fig.suptitle(
                f"layer: {layer_label}, category: {category_label}, image: {paths[image_idx]} ",
                fontsize="large",
            )
        # TODO add original image to the visualization and
        offset += len(grp)

        # Saving the preview
        if output_dir_path is not None:
            # Ensure that output directory exists if not create it
            if not os.path.exists(output_dir_path):
                # Create the directory
                os.mkdir(output_dir_path)

            if os.path.isdir(output_dir_path):
                plt.savefig(
                    gen_preview_output_file(
                        f"{layer_label}_{category}",
                        output_dir_path,
                        output_ext=output_ext,
                    )
                )
            else:
                LOGGER.warning(f"{output_dir_path} exists and is not a directory")

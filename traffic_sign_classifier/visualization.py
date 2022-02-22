""" Image grid visualization tools
@author Wajih Ouertani
@email wajih.ouertani@gmail.com
"""

import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from traffic_sign_classifier.utils import group_by_category


def show_image(ax, image: np.ndarray, title: Optional[str] = None, **kwargs):
    """a helper to show/plot image in a subplot (AxesSubplot)"""
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(image, **kwargs)
    if title:
        ax.set_title(f"{title}", pad=-2)


def gen_preview_output_file(
    index: int, output_dir_path: str, output_ext: str = "jpg"
) -> str:
    """Generate output file for a group of categories"""
    output_file_path = ".".join(["_".join(["preview", str(index)]), output_ext])
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
                """TODO: log saving error if the output_dir_path does not exist or could not be created"""

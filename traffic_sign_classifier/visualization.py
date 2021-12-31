from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from traffic_sign_classifier.utils import group_by_category


def grid_visu(
    images: np.ndarray,
    labels: List[int],
    sample_size: int = 3,
    categories: Optional[List[int]] = None,
    label_to_name: Optional[Dict[int, str]] = None,
    shuffle: bool = False,
    categories_per_fig: int = 5,  # number of category per figure
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

    # Group the images by category/class
    by_category = group_by_category(labels, sample_size=sample_size, shuffle=shuffle)

    # Optionally filter the categories (to visualize if categories is set in param)
    if categories is not None:
        by_category = {
            key: by_category[key]
            for key in set(categories).intersection(by_category.keys())
        }

    sorted_categories = sorted(by_category.keys())  # sort selected categories

    # group categories in separate figures (seemd loading faster)
    for grp in [
        sorted_categories[i : i + categories_per_fig]
        for i in range(0, len(sorted_categories), categories_per_fig)
    ]:

        subfigs = plt.figure(
            constrained_layout=True, figsize=(2 * sample_size, 2 * len(grp))
        ).subfigures(len(grp), 1, wspace=0.01)

        for category, fig in zip(grp, subfigs):
            # TODO: dynamically reshape the grid
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, len(by_category[category])),
                axes_pad=0.3,
            )
            for ax, idx in zip(grid, by_category[category]):
                if idx is not None:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.imshow(images[idx, :])
                    # ax.set_title(f"{idx}")
            fig.suptitle(f"{label_to_name.get(category, category)}", fontsize="large")

"""Create colormaps that allow mapping semantic segmentation outputs to color images"""

import numpy as np


def create_cityscapes_label_colormap():
    """
    Creates a label colormap used in CITYSCAPES segmentation benchmark.
    :return: A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]  # road
    colormap[1] = [244, 35, 232]  # sidewalk
    colormap[2] = [70, 70, 70]  # building
    colormap[3] = [102, 102, 156]  # wall
    colormap[4] = [190, 153, 153]  # fence
    colormap[5] = [153, 153, 153]  # pole
    colormap[6] = [250, 170, 30]  # traffic light
    colormap[7] = [220, 220, 0]  # traffic sign
    colormap[8] = [107, 142, 35]  # vegetation
    colormap[9] = [152, 251, 152]  # terrain
    colormap[10] = [70, 130, 180]  # sky
    colormap[11] = [220, 20, 60]  # person
    colormap[12] = [255, 0, 0]  # rider
    colormap[13] = [0, 0, 142]  # car
    colormap[14] = [0, 0, 70]  # truck
    colormap[15] = [0, 60, 100]  # bus
    colormap[16] = [0, 80, 100]  # train
    colormap[17] = [0, 0, 230]  # motorcycle
    colormap[18] = [119, 11, 32]  # bicycle
    return colormap


def create_carla_label_colormap():
    """
    Creates a label colormap used to display segmentation maps in Carla simulations.
    :return: A colormap for visualizing Carla segmentation maps.
    """
    colormap = np.zeros((13, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # not labeled
    colormap[1] = [70, 70, 70]  # building
    colormap[2] = [190, 153, 153]  # fence
    colormap[3] = [250, 170, 160]  # other
    colormap[4] = [220, 20, 60]  # pedestrain
    colormap[5] = [153, 153, 153]  # pole
    colormap[6] = [157, 234, 50]  # road line
    colormap[7] = [128, 64, 128]  # road
    colormap[8] = [244, 35, 232]  # side walk
    colormap[9] = [107, 142, 35]  # vegetation
    colormap[10] = [0, 0, 142]  # car
    colormap[11] = [102, 102, 156]  # wall
    colormap[12] = [220, 220, 0]  # traffic sign
    return colormap


def create_carla_label_colormap_cityscapes_style():
    """
    Creates a label colormap that maps Carla segmentation classes to the same color codes that Cityscapes uses.
    For most classes, this is already the case in create_carla_label_colormap().
    However, in this function here we return a strict subset of create_cityscapes_label_colormap().
    This means we map: road line > road, not labeled > terrain, other > terrain
    :return: A colormap for visualizing Carla segmentation maps with Cityscapes color scheme.
    """
    colormap = create_carla_label_colormap()
    colormap[0] = [152, 251, 152]  # not labeled > terrain
    colormap[3] = [152, 251, 152]  # other > terrain
    colormap[6] = [128, 64, 128]  # road line > road
    return colormap

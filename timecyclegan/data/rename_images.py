"""Rename all .png images in a folder to 000000.png, 0000001.png, ..."""

import os
from shutil import move

import numpy as np


def rename_images(dir_):
    files = [f for f in os.listdir(dir_) if f.endswith(".png")]
    print(len(files))
    num_digits = int(np.ceil(np.log10(len(files) - 1)))
    print(num_digits)
    assert num_digits == 4
    for i, filename in enumerate(sorted(files)):
        old_path = os.path.join(dir_, filename)
        new_path = os.path.join(dir_, str(i).zfill(num_digits) + ".png")
        move(old_path, new_path)


if __name__ == '__main__':
    rename_images(os.getcwd())

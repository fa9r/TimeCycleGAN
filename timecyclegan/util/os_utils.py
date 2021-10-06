"""OS Util functions"""

import os


def make_dir(dir_name):
    """
    Create a directory if it does not exist
    :param dir_name: Path of directory to be created
    :return: dir_name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

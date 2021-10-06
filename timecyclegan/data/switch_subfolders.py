"""Script to switch the subfolder hierarchy (<current_dir>/a/b/... will be changed to <current_dir>/b/a/...)"""

import os
from shutil import move


def main():
    """Main function"""
    sub_dir_names = sorted(os.listdir(os.curdir))
    for sub_dir_name in sub_dir_names:
        sub_dir = os.path.join(os.curdir, sub_dir_name)
        sub_sub_dir_names = sorted(os.listdir(sub_dir))
        for sub_sub_dir_name in sub_sub_dir_names:
            src = os.path.join(os.curdir, sub_dir_name, sub_sub_dir_name)
            dst = os.path.join(os.curdir, sub_sub_dir_name, sub_dir_name)
            move(src, dst)
        os.rmdir(sub_dir)


if __name__ == '__main__':
    main()

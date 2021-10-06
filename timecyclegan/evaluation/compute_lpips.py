"""
Calculate average LPIPS
Modified version of PerceptualSimilarity/compute_dists_dirs.py
"""

import argparse
import os
from timecyclegan.evaluation.PerceptualSimilarity.models import PerceptualLoss
from timecyclegan.evaluation.PerceptualSimilarity.util.util import (
    im2tensor,
    load_image,
)


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir0', help="First image directory.")
    parser.add_argument('dir1', help="Second image directory.")
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument(
        '--use_gpu', action='store_true', help='turn on flag to use GPU'
    )
    return parser


def calculate_lpips(dir0, dir1, version=0.1, use_gpu=False):
    model = PerceptualLoss(
        model='net-lin',
        net='alex',
        use_gpu=use_gpu,
        version=version,
    )
    dist = 0
    count = 0

    for root, _, fnames in sorted(os.walk(dir0)):
        for fname in sorted(fnames):
            path0 = os.path.join(root, fname)
            rel_path = os.path.relpath(path0, dir0)
            path1 = os.path.join(dir1, rel_path)
            # Load images
            img0 = im2tensor(load_image(path0))
            img1 = im2tensor(load_image(path1))
            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
            # Compute distance
            dist += model.forward(img0,img1).item()
            count += 1
    return dist/count


def main():
    parser = build_argparser()
    args = parser.parse_args()
    lpips = calculate_lpips(
        dir0=args.dir0,
        dir1=args.dir1,
        version=args.version,
        use_gpu=args.use_gpu,
    )
    print("LPIPS:", lpips)


if __name__ == '__main__':
    main()

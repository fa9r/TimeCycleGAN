"""
Compute FID score
Adapted from pytorch_fid/fid_score.py
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from timecyclegan.evaluation.pytorch_fid.inception import InceptionV3
from timecyclegan.evaluation.pytorch_fid.fid_score import calculate_fid_given_paths


def build_argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files'
    )
    parser.add_argument(
        '--batch-size', type=int, default=50, help='Batch size to use'
    )
    parser.add_argument(
        '--dims', type=int, default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help='Dimensionality of Inception features to use. '
             'By default, uses pool3 features'
    )
    parser.add_argument(
        '-c', '--gpu', default='', type=str,
        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(
        args.path,
        args.batch_size,
        args.gpu != '',
        args.dims,
        args.verbose
    )
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()

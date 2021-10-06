from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from timecyclegan.evaluation.pytorch_fid.inception import InceptionV3
from timecyclegan.evaluation.pytorch_fid.fid_score import calculate_fid_given_paths
from timecyclegan.evaluation.compute_tlp import calculate_tlp
from timecyclegan.evaluation.compute_lpips import calculate_lpips
from timecyclegan.evaluation.compute_tof import calculate_tof
from timecyclegan.util.logging import log_command, log_metrics


def build_argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('real_dir', help="Directory of real images.")
    parser.add_argument('fake_dir', help="Directory of fake images.")
    parser.add_argument("model_name", help="Name of the model/experiment.")
    parser.add_argument('-v', '--lpips_version', type=str, default='0.1')
    parser.add_argument(
        '--use_gpu', "-g", action='store_true', help='Set to use GPU.'
    )
    parser.add_argument(
        '--fid_batch_size', type=int, default=50, help='Batch size to use.'
    )
    parser.add_argument(
        '--dims', type=int, default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help='Dimensionality of Inception features to use. '
             'By default, uses pool3 features.'
    )
    parser.add_argument(
        "--height", "-ih", type=int, default=64,
        help="Resize images to have image_height."
    )
    parser.add_argument(
        "--width", "-iw", type=int, default=64,
        help="Resize images to have image_width."
    )
    parser.add_argument(
        "--unpaired_source_dir", "-u",
        help="Directory of source images which can be used to compute temporal"
             " metrics in unpaired generation. If set, LPIPS will not be"
             " computed and real_dir will only be used for FID."
    )
    return parser


def compute_metrics(
        real_dir,
        fake_dir,
        model_name,
        lpips_version="0.1",
        use_gpu=False,
        fid_batch_size=50,
        dims=2048,
        height=64,
        width=64,
        unpaired_source_dir=None
):
    """Compute FID, LPIPS, tLP, tOF"""
    fid = calculate_fid_given_paths(
        paths=(real_dir, fake_dir),
        batch_size=fid_batch_size,
        cuda=use_gpu,
        dims=dims,
        verbose=True,
    )
    print('FID: ', fid)
    if unpaired_source_dir is None:
        lpips = calculate_lpips(
            dir0=real_dir,
            dir1=fake_dir,
            version=lpips_version,
            use_gpu=use_gpu,
        )
        print("LPIPS:", lpips)
    else:
        lpips = 0
        real_dir = unpaired_source_dir
    tlp = calculate_tlp(
        dir0=real_dir,
        dir1=fake_dir,
        version=lpips_version,
        use_gpu=use_gpu,
        width=width,
        height=height,
    )
    print("tLP:", tlp)
    tof = calculate_tof(
        dir0=real_dir,
        dir1=fake_dir,
        width=width,
        height=height,
    )
    print("tOF:", tof)
    log_metrics(model_name, fid, lpips, tlp, tof)


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    log_command()
    compute_metrics(**vars(args))

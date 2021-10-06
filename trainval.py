from timecyclegan.util.argparser import parse_args
from timecyclegan.util.logging import log_command
from train import train
from val import validate


def trainval(**kwargs):
    """Train and validate a model: train > inference > metric computation"""
    print("*** TRAINING ***")
    train(**kwargs)
    print("*** VALIDATION ***")
    validate(**kwargs)


if __name__ == '__main__':
    trainval_kwargs = parse_args(mode="trainval")
    log_command()
    trainval(**trainval_kwargs)

from timecyclegan.models import get_task
from timecyclegan.util.argparser import parse_args
from timecyclegan.util.logging import log_command
from timecyclegan.evaluation.compute_metrics import compute_metrics
from test import test, define_output_dir


def validate(**kwargs):
    """Train and validate a model: train > inference > metric computation"""

    def get_fake_dir(dir_name_appendix=""):
        """Wrapper for define_output_dir to reduce code duplicates"""
        return define_output_dir(
            model_name=kwargs["model_name"],
            output_root=kwargs["test_output_dir"],
            dir_name_appendix=dir_name_appendix,
        )

    def compute_val_metrics(unpaired=False, source_to_target=False):
        """Compute metrics depending on task"""
        if unpaired:
            if source_to_target:
                real_dir = kwargs["test_target_dir"]
                fake_dir = get_fake_dir("source_to_target")
                unpaired_source_dir = kwargs["test_source_dir"]
            else:
                real_dir = kwargs["test_source_dir"]
                fake_dir = get_fake_dir("target_to_source")
                unpaired_source_dir = kwargs["test_target_dir"]
        else:
            real_dir = kwargs["test_target_dir"]
            fake_dir = get_fake_dir()
            unpaired_source_dir = None
        return compute_metrics(
            real_dir=real_dir,
            fake_dir=fake_dir,
            model_name=kwargs["model_name"],
            use_gpu=(kwargs["gpu"] >= 0),
            height=kwargs["image_height"],
            width=kwargs["image_width"],
            unpaired_source_dir=unpaired_source_dir,
        )

    print("*** INFERENCING MODEL ***")
    if get_task(kwargs["model_type"]) != "unpaired":
        test(**kwargs)
    else:
        kwargs["test_unpaired_target_to_source"] = False
        test(**kwargs)
        kwargs["test_unpaired_target_to_source"] = True
        test(**kwargs)

    print("*** COMPUTING METRICS ***")
    if get_task(kwargs["model_type"]) != "unpaired":
        compute_val_metrics()
    else:
        compute_val_metrics(unpaired=True, source_to_target=True)
        compute_val_metrics(unpaired=True, source_to_target=False)


if __name__ == '__main__':
    val_kwargs = parse_args(mode="val")
    log_command()
    validate(**val_kwargs)

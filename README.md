# TimeCycleGAN: A Meta-Architecture for Temporally Consistent GANs

![TimeCycleGAN Teaser](teaser.gif)

PyTorch meta-architecture for temporally consistent video generation with GANs.
Can be added to *any* existing PyTorch GAN model.
We provide example models for unconditional video generation, paired video-to-video translation, and unpaired video-to-video translation.
See [http://fa9r.com/files/timecyclegan_small.pdf][thesis] for technical details.

## Requirements
* Linux or macOS
* Python 3.7.4
* (Optional) NVIDIA GPU + CUDA 10.1

### Required Packages
* numpy 1.17.3
* Pillow 6.2.1
* tensorboard 2.0.0
* torch 1.3.0
* torchvision 0.4.1
* GitPython 3.1.0

### Additional Requirements for Metric Computations and Optical Flow
* matplotlib 3.1.1
* opencv-python 4.2.0.32
* scikit-image 0.17.2
* scipy 1.4.1
* tqdm 4.43.0
* pytz==2019.3
* scikit-image 0.17.2
* scipy 1.4.1

## Setup
### Installation
1. Make sure you have all required packages listed above or install with `pip install -r requirements.txt`.
2. Run `pip install -e .` to install the TimeCycleGAN code.
This will allow you to run the scripts from any location
and it also allows you to import TimeCycleGAN in your own projects
with e.g. `from timecyclegan import TimeCycleMixin`.
3. If you want to use models or metrics with optical flow, you need to install
 [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch/) first via
`cd timecyclegan/models/networks/flownet2_pytorch/; bash install.sh`.
Note that this will require a NVIDIA GPU with CUDA.

**Full Install:**
```
pip install -r requirements.txt
pip install -e .
cd timecyclegan/models/networks/flownet2_pytorch/
bash install.sh
```

## Repository Structure
- `timecyclegan/` is the sources root.
- `timecyclegan/data/` contains scripts for generating/modifying/preparing
Cityscapes/CARLA data.
- `timecyclegan/datasets/` contains subclasses of `torch.utils.data.Dataset`
that we used for all experiments.
- `timecyclegan/evaluation/` contains evaluation code to compute
FID, LPIPS, tLP, and tOF metrics.
- `timecyclegan/models/` contains all GAN models.
All models inherit from the abstract classes in 
`timecyclegan/models/base_model.py`.
Use `get_model()` in `timecyclegan/models/__init__.py` to load a specific
model.
- `timecyclegan/models/networks/` contains network architecture defintions
of all discriminators, generators, and others.
- `timecyclegan/models/baselines/` contains reimplementation of all
baseline models (pix2pix, CycleGAN, RecycleGAN, vid2vid, DCGAN).
- `timecyclegan/models/sequential/` contains modified versions with
recurrent generators for all non-sequential baselines 
(pix2pix, CycleGAN, RecycleGAN, DCGAN).
We subclass those in order to build our TimeCycleGAN models shown in the
paper (`timecycle_pix2pix.py` and `timecycle_cyclegan.py` in particular).
- `timecyclegan/models/mixins/` contains Python Mixins for easily adding
the TimeCycle loss, sequence discriminators, or a warp loss to your models.
See Usage chapter below.

## TimeCycleGAN Models (from my Thesis)
In my thesis, four specific TimeCycleGAN models are explained. Those
correspond to code classes as follows:
- **TimeCycleGAN-U** is `timecyclegan.models.timecycle_dcgan.TimeCycleDCGAN`
- **TimeCycleGAN-UP** is `timecyclegan.models.timecycle_cyclegan.TimeCycleCycleGAN`
- **TimeCycleGAN-P** and **TimeCycleGAN-P++** are `timecyclegan.models.timecycle_pix2pix.TimeCyclePix2Pix`
and `timecyclegan.models.timecycle_pix2pix.TimeCyclePix2Pix` with different
hyperparameter choices.

See chapter 4.1.2. on page 34 in [my thesis][thesis] for a detailed list of hyperparameters.

## Adding TimeCycle Loss to a GAN
You can add the TimeCycle loss to your own GAN in as few as 10 LoC.
Assuming your GAN is implemented in PyTorch as a class with separate
methods for forward/backward behavior, simply subclass your GAN
and implement the TimecycleMixin as shown below:
```python
from timecyclegan import TimeCycleMixin

class MyGANWithTimecycle(TimeCycleMixin, MyGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # init your GAN and TimecycleMixin
        self.define_timecycle(...)
        self.add_timecycle_discriminator(...) # optional

def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs) # perform forward() of your GAN
        self.timecycle_forward(...)
        self.set_timecycle_discriminator_input(...) # optional

def backward(self, *args, **kwargs):
        super().backward(*args, **kwargs) # perform backward() of your GAN
        self.loss_G += self.timecycle_backward(...)
        self.loss_D += self.timecycle_backward_D(...) # optional
```

For usage examples, see my custom models `timecyclegan/models/timecycle_<x>.py`.
They also show how you can make use of my other mixins for 
sequence disrciminators and warp losses.

For a more detailed usage guide, see chapter 4.2. on page 35 of [my thesis][thesis].

## Usage

### Training
* To train a model, run `python train.py <your_experiment_name> <source_directory> <target_directory> [optional args]`
* See `python train.py -h` for a full list of optional args or check `timecyclegan/util/argparser.py`.
* Training logs will be written to `runs/<your_experiment_name>`. See Logging section below for usage.
* Checkpoints and saved models will be written to `checkpoints/<your_experiment_name>`.

### Validation
* Use `python val.py <your_experiment_name> <val_source_dir> <val_target_dir> [optional args]`
to inference a trained model with name `<your_experiment_name>` on the given validation dataset
and to compute FID, LPIPS, tLP and tOF metrics on it.
* See `python val.py -h` for a list of optional args or check `timecyclegan/util/argparser.py`.
* By default, result images will be written to `results/<your_experiment_name>` (customizable with `-to` arg).
* Validation metrics are displayed in the terminal and will also be logged to `metrics.csv`.

### Training + Validation
Training and validation can be done in one using `python trainval.py` while supplying both train and val arguments.

### Testing
To inference a model, use `python test.py` with same args as for validation.
This will do the same as `val.py` just without the metric computation.

### Logging
* Training losses, images, hparams will be logged to Tensorboard. Use `tensorboard --logdir=runs/<your_experiment_name>` to inspect.
* Hyperparameters of all trained models are also logged to `train_log.csv` together with current git commit hash.
* Metrics computed during validation will be logged to `metrics.csv`.
* All commands you run are logged to `command_log.txt`.

## Citation
If you use this code in your research, please cite my Master's thesis:
```
@unpublished{TimeCycleGAN2020,
  title={Learning Temporal Consistency in Video Generation},
  author={Altenberger, Felix and Niessner, Matthias},
  year={2020},
}
```

## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
from which we adapt all generator/discriminator network architectures for paired and unpaired video-to-video translation models.

Optical flow estimation is performed by [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch/).

[pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
are used for FID and LPIPS metric computations.

[thesis]: http://fa9r.com/files/timecyclegan_small.pdf
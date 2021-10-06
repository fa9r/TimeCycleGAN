import torch

from timecyclegan.models.base_model import BaseModel, PairedModel, UnpairedModel
from timecyclegan.util.dict_utils import init_dicts, pack_dicts, unpack_dicts

# pylint: disable=abstract-method


class TimeCycleMixin(BaseModel):
    """
    Mixin that defines all Timecycle functionality of TimecycleGAN
    Can be used in any model that inherits from BaseModel (certain BaseModel functionality is required!)
    """
    def __init__(
            self,
            timecycle_type="conditional",
            timecycle_loss="l1",
            timecycle_loss_weight=0,
            timecycle_motion_model_architecture='resnet_1blocks',
            timecycle_motion_model_filters=64,
            timecycle_separate_motion_models=False,
            timecycle_warp_loss_weight=0,
            **kwargs
    ):
        """
        :param timecycle_loss_weight: relative weight of the timecycle loss compared to the GAN loss
        :param timecycle_motion_model_architecture: architecture of the motion model
        :param timecycle_motion_model_filters: number of filters in the last conv layer of the motion model
        :param timecycle_names: Names of all timecycles, in case multiple are to be used
        :param timecycle_separate_motion_models: If True, use separate motion models for forward/backward predictions
        """
        super().__init__(**kwargs)
        if timecycle_type not in ("conditional", "unconditional", "pingpong", "timerecycle"):
            raise ValueError("Invalid argument: timecycle_type '%s' not recognized." % timecycle_type)
        if timecycle_loss == "l1":
            self.criterion_timecycle = torch.nn.L1Loss()
        elif timecycle_loss == "l2":
            self.criterion_timecycle = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid argument: timecycle_loss '%s' not recognized." % timecycle_loss)
        self.hparam_dict["Timecycle"] = True
        self.timecycle_type = timecycle_type
        self.hparam_dict["Timecycle Type"] = timecycle_type
        self.hparam_dict["Timecycle Loss"] = timecycle_loss
        self.timecycle_loss_weight = timecycle_loss_weight
        self.hparam_dict["Timecycle Loss Weight"] = timecycle_loss_weight
        self.timecycle_motion_model_architecture = timecycle_motion_model_architecture
        self.hparam_dict["Timecycle Motion Model"] = timecycle_motion_model_architecture
        self.timecycle_motion_model_filters = timecycle_motion_model_filters
        self.hparam_dict["Timecycle Motion Model Filters"] = timecycle_motion_model_filters
        self.timecycle_separate_motion_models = timecycle_separate_motion_models
        self.hparam_dict["Timecycle Separate Motion Models"] = timecycle_separate_motion_models
        self.timecycle_warp_loss_weight = timecycle_warp_loss_weight
        self.hparam_dict["Timecycle Warp Loss Weight"] = timecycle_warp_loss_weight
        self._default_timecycle_name = "target"
        self.__default_discriminator_name = ""
        self.__num_losses_by_type = {
            "conditional": 4,
            "unconditional": 2,
            "pingpong": 1,
        }
        self.timecycle_names = []  # set in define_timecycle()
        self._loss_G_timecycle = None  # set in backward_G()
        self._loss_D_timecycle = None  # set in backward_D()
        # initialize various attributes as empty dicts {timecycle_name: value}
        self._timecycle_len = {}  # set in timecycle_forward()
        self.__discriminator_names = {}
        self.__motion_model_forward = {}  # set in timecycle_define_motion_models()
        self.__motion_model_backward = {}  # set in timecycle_define_motion_models()
        self.__D_frame, self.__D_seq = init_dicts(2)  # set in timecycle_define_discriminator()
        self.__discriminator_seqs = {}  # set in set_timecycle_discriminator_input()
        self.__sequence, self.__conditioning_sequence = init_dicts(2)  # set in timecycle_forward()
        self.__pred_long, self.__pred_skip = init_dicts(2)  # set in timecycle_forward()
        self.__rec_long, self.__rec_skip = init_dicts(2)  # set in timecycle_forward()
        self.__pred_rec_dicts = [self.__pred_long, self.__pred_skip, self.__rec_long, self.__rec_skip]
        self.__loss_dict_G = {}  # set in backward_G()
        self.__loss_dicts_G_timecycle = init_dicts(5)  # set in timecycle_backward_timecycle()
        self.__loss_dicts_G_frame = init_dicts(5)  # set in timecycle_backward_G_frame()
        self.__loss_dicts_G_seq = init_dicts(3)  # set in timecycle_backward_G_seq()
        self.__loss_dict_D = {}  # set in backward_D()
        self.__loss_dicts_D_frame = init_dicts(7)  # set in timecycle-backward_D_frame()
        self.__loss_dicts_D_seq = init_dicts(5)  # set in timecycle_backward_D_seq()
        self.__warp_loss = {}

    def __set_name(self, timecycle_name):
        timecycle_name = timecycle_name if timecycle_name is not None else self._default_timecycle_name
        assert timecycle_name not in self.timecycle_names
        self.timecycle_names.append(timecycle_name)

    def _check_timecycle_name(self, timecycle_name):
        """Check if given timecycle_name is valid and set to default if not provided"""
        timecycle_name = timecycle_name if timecycle_name is not None else self._default_timecycle_name
        assert timecycle_name in self.timecycle_names
        return timecycle_name

    def __check_discriminator_name(self, timecycle_name, discriminator_name):
        assert timecycle_name in self.__discriminator_names
        discriminator_name = discriminator_name if discriminator_name is not None else self.__default_discriminator_name
        assert discriminator_name in self.__discriminator_names[timecycle_name]
        return discriminator_name

    def __set_motion_model(self, motion_model, forward=True, save_model=True, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        motion_model_dict = self.__motion_model_forward if forward else self.__motion_model_backward
        motion_model_dict[timecycle_name] = motion_model
        if save_model:
            motion_model_name = "timecycle_%s_motion_model_%s" % (timecycle_name, "forward" if forward else "backward")
            self.generators[motion_model_name] = motion_model

    def __define_motion_model(self, input_channels, output_channels):
        return self.define_G(
            input_channels=input_channels,
            output_channels=output_channels,
            architecture=self.timecycle_motion_model_architecture,
            filters=self.timecycle_motion_model_filters,
        )

    def define_timecycle(self, input_channels, output_channels, timecycle_name=None):
        assert self.timecycle_type != "pingpong"
        self.__set_name(timecycle_name)

        def get_motion_model():
            return self.__define_motion_model(input_channels, output_channels)

        if self.timecycle_separate_motion_models:
            motion_model_forward, motion_model_backward = get_motion_model(), get_motion_model()
        else:
            motion_model_forward = motion_model_backward = get_motion_model()
        self.__set_motion_model(motion_model_forward, forward=True, timecycle_name=timecycle_name)
        self.__set_motion_model(motion_model_backward, forward=False, timecycle_name=timecycle_name)

    def define_timecycle_pingpong(self, pingpong_generator, timecycle_name=None):
        assert self.timecycle_type == "pingpong"
        self.__set_name(timecycle_name)
        self.__set_motion_model(pingpong_generator, forward=False, save_model=False, timecycle_name=timecycle_name)

    def define_timecycle_timerecyclev2(self, predictor, timecycle_name=None):
        assert self.timecycle_type == "unconditional"
        self.__set_name(timecycle_name)
        self.__set_motion_model(predictor, forward=True, save_model=False, timecycle_name=timecycle_name)
        self.__set_motion_model(predictor, forward=False, save_model=False, timecycle_name=timecycle_name)

    def add_timecycle_discriminator(
            self, discriminator, n_frames=1, step=1, discriminator_name=None, timecycle_name=None
    ):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        # add discriminator name to self.__discriminator_names[timecycle_name]
        discriminator_name = discriminator_name if discriminator_name is not None else self.__default_discriminator_name
        if timecycle_name not in self.__discriminator_names:
            self.__discriminator_names[timecycle_name] = []
        if discriminator_name not in self.__discriminator_names[timecycle_name]:
            self.__discriminator_names[timecycle_name].append(discriminator_name)
        # frame discriminator
        if n_frames == 1:
            if timecycle_name not in self.__D_frame:
                self.__D_frame[timecycle_name] = {}
            assert discriminator_name not in self.__D_frame[timecycle_name]
            self.__D_frame[timecycle_name][discriminator_name] = discriminator
        # sequence discriminator
        else:
            if timecycle_name not in self.__D_seq:
                self.__D_seq[timecycle_name] = {}
            assert discriminator_name not in self.__D_seq[timecycle_name]
            self.__D_seq[timecycle_name][discriminator_name] = discriminator

    def set_timecycle_discriminator_input(
            self, real_sequence, conditioning_sequence=None, ignore_first_n=0,
            discriminator_name=None, timecycle_name=None,
    ):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        real_sequence = real_sequence[ignore_first_n:]
        if conditioning_sequence is not None:
            conditioning_sequence = conditioning_sequence[ignore_first_n:]
        if timecycle_name not in self.__discriminator_seqs:
            self.__discriminator_seqs[timecycle_name] = {}
        self.__discriminator_seqs[timecycle_name][discriminator_name] = (real_sequence, conditioning_sequence)

    def __get_discriminator_input(self, discriminator_name=None, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        return self.__discriminator_seqs[timecycle_name][discriminator_name]

    def timecycle_forward(self, sequence, conditioning_sequence=None, ignore_first_n=0, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        sequence = sequence[ignore_first_n:]
        conditioning_sequence = conditioning_sequence[ignore_first_n:] if conditioning_sequence is not None else None
        # predict backward cycles: t -> t-i -> t for all i in (1, ..., B-1)
        preds_long, preds_skip, recs_long, recs_skip = [], [], [], []
        if self.timecycle_type != "pingpong":
            motion_model_forward = self.__motion_model_forward[timecycle_name]
        motion_model_backward = self.__motion_model_backward[timecycle_name]
        if self.timecycle_type in ("conditional", "pingpong"):
            self._timecycle_len[timecycle_name] = timecycle_len = len(sequence) - 1
            if timecycle_len > 0:
                pred_long = sequence[-1]
                for i in range(timecycle_len):
                    current_index = timecycle_len - 1 - i
                    current_input = conditioning_sequence[current_index]
                    # long cycle prediction
                    pred_long = motion_model_backward(torch.cat((current_input, pred_long), 1))
                    preds_long.append(pred_long)
                    if self.timecycle_type == "conditional":
                        # skip cycle prediction
                        pred_skip = motion_model_backward(torch.cat((current_input, sequence[-1]), 1))
                        preds_skip.append(pred_skip)
                        # skip cycle reconstruction
                        recs_skip.append(motion_model_forward(torch.cat((conditioning_sequence[-1], pred_skip), 1)))
                        # long cycle reconstruction
                        rec_long = pred_long
                        for j in range(i + 1):
                            next_input = conditioning_sequence[current_index + 1 + j]
                            rec_long = motion_model_forward(torch.cat((next_input, rec_long), 1))
                        recs_long.append(rec_long)
        elif self.timecycle_type == "unconditional":
            self._timecycle_len[timecycle_name] = timecycle_len = len(sequence) - 2
            if timecycle_len > 0:
                preds_long = [sequence[-1], sequence[-2]]
                for i in range(timecycle_len):
                    preds_long.append(motion_model_backward(torch.cat(preds_long[-2:], 1)))
                    recs_tmp = [preds_long[-1], preds_long[-2]]
                    for j in range(i + 1):
                        recs_tmp.append(motion_model_forward(torch.cat(recs_tmp[-2:], 1)))
                    recs_long.append(recs_tmp[-1])
                preds_long = preds_long[2:]
        # reverse preds/recs to be in chronological order (t-B+1, ..., t-1) and set attributes
        reversed_preds_recs = [seq[::-1] for seq in (preds_long, preds_skip, recs_long, recs_skip)]
        pack_dicts(
            dicts=[self.__sequence, self.__conditioning_sequence, *self.__pred_rec_dicts],
            key=timecycle_name,
            values=[sequence, conditioning_sequence, *reversed_preds_recs],
        )

    def timecycle_backward(self, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        loss_G_cycle_long, loss_G_cycle_skip, loss_G_temp_long, loss_G_temp_skip = [0] * 4
        # unpack sequences
        sequence, preds_long, preds_skip, recs_long, recs_skip = unpack_dicts(
            dicts=[self.__sequence, *self.__pred_rec_dicts],
            key=timecycle_name,
        )
        # calculate losses
        timecycle_len = self._timecycle_len[timecycle_name]
        for i in range(timecycle_len):
            loss_G_temp_long += self.criterion_timecycle(preds_long[i], sequence[i])
            if self.timecycle_type == "conditional":
                loss_G_temp_skip += self.criterion_timecycle(preds_skip[i], sequence[i])
                loss_G_cycle_skip += self.criterion_timecycle(recs_skip[i], sequence[-1])
            if self.timecycle_type != "pingpong":
                loss_G_cycle_long += self.criterion_timecycle(recs_long[i], sequence[-1])
        losses = [loss_G_cycle_long, loss_G_cycle_skip, loss_G_temp_long, loss_G_temp_skip]
        loss_G_cycle_temp = sum(losses) / self.__num_losses_by_type[self.timecycle_type]
        losses = [*losses, loss_G_cycle_temp]
        # divide all losses by timecycle_len and set attributes
        pack_dicts(
            dicts=self.__loss_dicts_G_timecycle,
            key=timecycle_name,
            values=[loss / timecycle_len for loss in losses] if timecycle_len > 0 else [0] * len(losses)
        )
        return (loss_G_cycle_temp / timecycle_len) * self.timecycle_loss_weight if timecycle_len > 0 else 0

    def timecycle_backward_warp(self, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        sequence, conditioning_sequence = unpack_dicts(
            dicts=[self.__sequence, self.__conditioning_sequence],
            key=timecycle_name,
        )
        if self.timecycle_type != "pingpong":
            motion_model_forward = self.__motion_model_forward[timecycle_name]
        motion_model_backward = self.__motion_model_backward[timecycle_name]
        warp_loss = 0
        timecycle_len = self._timecycle_len[timecycle_name]
        for i in range(timecycle_len):
            if self.timecycle_type == "conditional":
                target_forward = motion_model_forward(torch.cat((conditioning_sequence[i+1], sequence[i]), 1))
                warp_loss += self.criterion_L1(target_forward.detach(), sequence[i+1])
                target_backward = motion_model_backward(torch.cat((conditioning_sequence[i], sequence[i+1]), 1))
                warp_loss += self.criterion_L1(target_backward.detach(), sequence[i])
            elif self.timecycle_type == "unconditional":
                target_forward = motion_model_forward(torch.cat(sequence[i:i+2], 1))
                warp_loss += self.criterion_L1(target_forward.detach(), sequence[i+2])
                target_backward = motion_model_backward(torch.cat(sequence[i+1:i+3], 1))
                warp_loss += self.criterion_L1(target_backward.detach(), sequence[i])
        self.__warp_loss[timecycle_name] = warp_loss / timecycle_len if timecycle_len > 0 else 0
        return warp_loss * self.timecycle_warp_loss_weight

    def timecycle_backward_D_frame(self, discriminator_name=None, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        discriminator = self.__D_frame[timecycle_name][discriminator_name]
        loss_D_frame_real = 0
        loss_D_frame_pred_long, loss_D_frame_pred_skip, loss_D_frame_rec_long, loss_D_frame_rec_skip = [0] * 4
        # unpack sequences
        conditioning_sequence, preds_long, preds_skip, recs_long, recs_skip = unpack_dicts(
            dicts=[self.__conditioning_sequence, *self.__pred_rec_dicts],
            key=timecycle_name,
        )
        # define real images and conditioning for discriminator
        real_sequence, conditioning_sequence = self.__get_discriminator_input(discriminator_name, timecycle_name)
        # calculate losses
        timecycle_len = self._timecycle_len[timecycle_name]
        for i in range(timecycle_len):
            conditioning = conditioning_sequence[i] if conditioning_sequence is not None else None
            loss_D_frame_real += self.get_GAN_loss(real_sequence[i], discriminator, True, conditioning)
            loss_D_frame_pred_long += self.get_GAN_loss(preds_long[i].detach(), discriminator, False, conditioning)
            if self.timecycle_type == "conditional":
                loss_D_frame_pred_skip += self.get_GAN_loss(preds_skip[i].detach(), discriminator, False, conditioning)
                loss_D_frame_rec_skip += self.get_GAN_loss(recs_skip[i].detach(), discriminator, False, conditioning)
            if self.timecycle_type != "pingpong":
                loss_D_frame_rec_long += self.get_GAN_loss(recs_long[i].detach(), discriminator, False, conditioning)
        fake_losses = [loss_D_frame_pred_long, loss_D_frame_pred_skip, loss_D_frame_rec_long, loss_D_frame_rec_skip]
        loss_D_frame_fake = sum(fake_losses) / self.__num_losses_by_type[self.timecycle_type]
        loss_D_frame = (loss_D_frame_real + loss_D_frame_fake) / 2
        losses = [loss_D_frame, loss_D_frame_real, loss_D_frame_fake, *fake_losses]
        pack_dicts(
            dicts=self.__loss_dicts_D_frame,
            key=(timecycle_name, discriminator_name),
            values=[loss / timecycle_len for loss in losses] if timecycle_len > 0 else [0] * len(losses)
        )
        return loss_D_frame / timecycle_len if timecycle_len > 0 else 0

    def timecycle_backward_D_seq(self, discriminator_name=None, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        discriminator = self.__D_seq[timecycle_name][discriminator_name]
        # unpack dicts
        sequence, conditioning_sequence = unpack_dicts(
            dicts=[self.__sequence, self.__conditioning_sequence],
            key=timecycle_name,
        )
        pred_long, pred_skip = unpack_dicts(
            dicts=[self.__pred_long, self.__pred_skip],
            key=timecycle_name,
        )
        # define real images and conditioning for discriminator
        real_sequence, conditioning_sequence = self.__get_discriminator_input(discriminator_name, timecycle_name)
        # append last frame to preds, so preds have same length as the sequence, and concatenate all needed tensors
        pred_long = torch.cat([*pred_long, sequence[-1]], 1).detach()
        pred_skip = torch.cat([*pred_skip, sequence[-1]], 1).detach()
        loss_D_seq_real = self.get_GAN_loss(real_sequence, discriminator, True, conditioning_sequence)
        loss_D_seq_pred_long = self.get_GAN_loss(pred_long, discriminator, False, conditioning_sequence)
        loss_D_seq_pred_skip = self.get_GAN_loss(pred_skip, discriminator, False, conditioning_sequence)
        loss_D_seq_fake = (loss_D_seq_pred_long + loss_D_seq_pred_skip) / 2
        loss_D_seq = (loss_D_seq_real + loss_D_seq_fake) / 2
        losses = [loss_D_seq, loss_D_seq_real, loss_D_seq_fake, loss_D_seq_pred_long, loss_D_seq_pred_skip]
        pack_dicts(
            dicts=self.__loss_dicts_D_seq,
            key=(timecycle_name, discriminator_name),
            values=losses,
        )
        return loss_D_seq

    def timecycle_backward_G_frame(self, discriminator_name=None, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        discriminator = self.__D_frame[timecycle_name][discriminator_name]
        loss_G_GAN_frame_pred_long, loss_G_GAN_frame_pred_skip = [0] * 2
        loss_G_GAN_frame_rec_long, loss_G_GAN_frame_rec_skip = [0] * 2
        # unpack sequences
        conditioning_sequence, preds_long, preds_skip, recs_long, recs_skip = unpack_dicts(
            dicts=[self.__conditioning_sequence, *self.__pred_rec_dicts],
            key=timecycle_name,
        )
        # define real images and conditioning for discriminator
        _, conditioning_sequence = self.__get_discriminator_input(discriminator_name, timecycle_name)
        # calculate losses
        timecycle_len = self._timecycle_len[timecycle_name]
        for i in range(timecycle_len):
            conditioning = conditioning_sequence[i] if conditioning_sequence is not None else None
            loss_G_GAN_frame_pred_long += self.get_GAN_loss(preds_long[i], discriminator, conditioned_on=conditioning)
            if self.timecycle_type == "conditional":
                loss_G_GAN_frame_pred_skip += self.get_GAN_loss(preds_skip[i], discriminator, conditioned_on=conditioning)
                loss_G_GAN_frame_rec_skip += self.get_GAN_loss(recs_skip[i], discriminator, conditioned_on=conditioning)
            if self.timecycle_type != "pingpong":
                loss_G_GAN_frame_rec_long += self.get_GAN_loss(recs_long[i], discriminator, conditioned_on=conditioning)
        losses = [loss_G_GAN_frame_pred_long, loss_G_GAN_frame_pred_skip, loss_G_GAN_frame_rec_long, loss_G_GAN_frame_rec_skip]
        loss_G_GAN_frame = sum(losses) / self.__num_losses_by_type[self.timecycle_type]
        # divide all losses by timecycle_len and set attributes
        losses = [loss_G_GAN_frame, *losses]
        pack_dicts(
            dicts=self.__loss_dicts_G_frame,
            key=(timecycle_name, discriminator_name),
            values=[loss / timecycle_len for loss in losses] if timecycle_len > 0 else [0] * len(losses),
        )
        return loss_G_GAN_frame / timecycle_len if timecycle_len > 0 else 0

    def timecycle_backward_G_seq(self, discriminator_name=None, timecycle_name=None):
        timecycle_name = self._check_timecycle_name(timecycle_name)
        discriminator_name = self.__check_discriminator_name(timecycle_name, discriminator_name)
        discriminator = self.__D_seq[timecycle_name][discriminator_name]
        # unpack dicts
        sequence, conditioning_sequence = unpack_dicts(
            dicts=[self.__sequence, self.__conditioning_sequence],
            key=timecycle_name
        )
        pred_long, pred_skip = unpack_dicts(
            dicts=[self.__pred_long, self.__pred_skip],
            key=timecycle_name
        )
        # define real images and conditioning for discriminator
        _, conditioning_sequence = self.__get_discriminator_input(discriminator_name, timecycle_name)
        # append last frame to preds, so preds have same length as the sequence, and concatenate all needed tensors
        pred_long = torch.cat([*pred_long, sequence[-1]], 1)
        pred_skip = torch.cat([*pred_skip, sequence[-1]], 1)
        # calculate losses and set attributes
        loss_G_GAN_seq_long = self.get_GAN_loss(pred_long, discriminator, conditioned_on=conditioning_sequence)
        loss_G_GAN_seq_skip = self.get_GAN_loss(pred_skip, discriminator, conditioned_on=conditioning_sequence)
        losses = [loss_G_GAN_seq_long, loss_G_GAN_seq_skip]
        loss_G_GAN_seq = sum(losses) / len(losses)
        pack_dicts(
            dicts=self.__loss_dicts_G_seq,
            key=(timecycle_name, discriminator_name),
            values=[loss_G_GAN_seq, *losses],
        )
        # average and return losses
        return loss_G_GAN_seq

    def backward_D(self):
        self._loss_D_timecycle = torch.tensor(0.0).to(self.device)
        for timecycle_name in self.timecycle_names:
            self.__loss_dict_D[timecycle_name] = torch.tensor(0.0).to(self.device)
            if timecycle_name in self.__D_frame:
                for discriminator_name in self.__D_frame[timecycle_name]:
                    self.__loss_dict_D[timecycle_name] += self.timecycle_backward_D_frame(discriminator_name, timecycle_name)
            if timecycle_name in self.__D_seq:
                for discriminator_name in self.__D_seq[timecycle_name]:
                    self.__loss_dict_D[timecycle_name] += self.timecycle_backward_D_seq(discriminator_name, timecycle_name)
            self._loss_D_timecycle += self.__loss_dict_D[timecycle_name]
        self.loss_D += self._loss_D_timecycle
        super().backward_D()

    def backward_G(self):
        self._loss_G_timecycle = torch.tensor(0.0).to(self.device)
        for timecycle_name in self.timecycle_names:
            self.__loss_dict_G[timecycle_name] = self.timecycle_backward(timecycle_name)
            self.__loss_dict_G[timecycle_name] += self.timecycle_backward_warp(timecycle_name)
            if timecycle_name in self.__D_frame:
                for discriminator_name in self.__D_frame[timecycle_name]:
                    self.__loss_dict_G[timecycle_name] += self.timecycle_backward_G_frame(discriminator_name, timecycle_name)
            if timecycle_name in self.__D_seq:
                for discriminator_name in self.__D_seq[timecycle_name]:
                    self.__loss_dict_G[timecycle_name] += self.timecycle_backward_G_seq(discriminator_name, timecycle_name)
            self._loss_G_timecycle += self.__loss_dict_G[timecycle_name]
        self.loss_G += self._loss_G_timecycle
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        timecycle_losses = {
            "Discriminator/Timecycle": self._loss_D_timecycle,
            "Generator/Timecycle": self._loss_G_timecycle
        }
        timecycle_images = {}
        loss_template_no_D = {
            'Discriminator/Timecycle/%s': self.__loss_dict_D,
            'Generator/Timecycle/%s': self.__loss_dict_G,
            'Generator/Timecycle/%s/Timecycle/Long': self.__loss_dicts_G_timecycle[0],
            'Generator/Timecycle/%s/Timecycle/Skip': self.__loss_dicts_G_timecycle[1],
            'Generator/Timecycle/%s/Timecycle/Long/Temp': self.__loss_dicts_G_timecycle[2],
            'Generator/Timecycle/%s/Timecycle/Skip/Temp': self.__loss_dicts_G_timecycle[3],
            'Generator/Timecycle/%s/Timecycle': self.__loss_dicts_G_timecycle[4],
            'Generator/Timecycle/%s/Warp': self.__warp_loss,
        }
        loss_template_D_frame = {
            'Discriminator/Timecycle/%s/Frame/%s': self.__loss_dicts_D_frame[0],
            'Discriminator/Timecycle/%s/Frame/%s/Real': self.__loss_dicts_D_frame[1],
            'Discriminator/Timecycle/%s/Frame/%s/Fake': self.__loss_dicts_D_frame[2],
            'Discriminator/Timecycle/%s/Frame/%s/Fake/Long/Pred': self.__loss_dicts_D_frame[3],
            'Discriminator/Timecycle/%s/Frame/%s/Fake/Skip/Pred': self.__loss_dicts_D_frame[4],
            'Discriminator/Timecycle/%s/Frame/%s/Fake/Long/Rec': self.__loss_dicts_D_frame[5],
            'Discriminator/Timecycle/%s/Frame/%s/Fake/Skip/Rec': self.__loss_dicts_D_frame[6],
            'Generator/Timecycle/%s/GAN/Frame/%s': self.__loss_dicts_G_frame[0],
            'Generator/Timecycle/%s/GAN/Frame/%s/Long/Pred': self.__loss_dicts_G_frame[1],
            'Generator/Timecycle/%s/GAN/Frame/%s/Skip/Pred': self.__loss_dicts_G_frame[2],
            'Generator/Timecycle/%s/GAN/Frame/%s/Long/Rec': self.__loss_dicts_G_frame[3],
            'Generator/Timecycle/%s/GAN/Frame/%s/Skip/Rec': self.__loss_dicts_G_frame[4],
        }
        loss_template_D_seq = {
            'Discriminator/Timecycle/%s/Sequence/%s': self.__loss_dicts_D_seq[0],
            'Discriminator/Timecycle/%s/Sequence/%s/Real': self.__loss_dicts_D_seq[1],
            'Discriminator/Timecycle/%s/Sequence/%s/Fake': self.__loss_dicts_D_seq[2],
            'Discriminator/Timecycle/%s/Sequence/%s/Fake/Long': self.__loss_dicts_D_seq[3],
            'Discriminator/Timecycle/%s/Sequence/%s/Fake/Skip': self.__loss_dicts_D_seq[4],
            'Generator/Timecycle/%s/GAN/Sequence/%s': self.__loss_dicts_G_seq[0],
            'Generator/Timecycle/%s/GAN/Sequence/%s/Long': self.__loss_dicts_G_seq[1],
            'Generator/Timecycle/%s/GAN/Sequence/%s/Skip': self.__loss_dicts_G_seq[2],
        }
        image_template = {
            'Timecycle/%s/Prediction/Long': self.__pred_long,
        }
        if self.timecycle_type == "conditional":
            image_template['Timecycle/%s/Prediction/Skip'] = self.__pred_skip
            image_template['Timecycle/%s/Reconstruction/Skip'] = self.__rec_skip
        if self.timecycle_type != "pingpong":
            image_template['Timecycle/%s/Reconstruction/Long'] = self.__rec_long
        for timecycle_name in self.timecycle_names:
            # add losses independent of discriminators
            losses = {
                loss_name % timecycle_name: loss[timecycle_name]
                for loss_name, loss in loss_template_no_D.items() if timecycle_name in loss
            }
            # add discriminator losses
            for discriminator_name in self.__discriminator_names[timecycle_name]:
                # add sequence discriminator losses
                if timecycle_name in self.__D_seq and discriminator_name in self.__D_seq[timecycle_name]:
                    losses_D_seq = {
                        loss_name % (timecycle_name, discriminator_name): loss[timecycle_name][discriminator_name]
                        for loss_name, loss in loss_template_D_seq.items() if timecycle_name in loss
                    }
                    losses = {**losses, **losses_D_seq}
                # add frame discriminator losses
                if timecycle_name in self.__D_frame and discriminator_name in self.__D_frame[timecycle_name]:
                    losses_D_frame = {
                        loss_name % (timecycle_name, discriminator_name): loss[timecycle_name][discriminator_name]
                        for loss_name, loss in loss_template_D_frame.items() if timecycle_name in loss
                    }
                    losses = {**losses, **losses_D_frame}
            # add timecycle images
            timecycle_len = self._timecycle_len[timecycle_name]
            images = {
                image_name % timecycle_name: (image[timecycle_name], {'nrow': timecycle_len})
                for image_name, image in image_template.items()
            } if timecycle_len > 0 else {}
            timecycle_losses = {**timecycle_losses, **losses}
            timecycle_images = {**timecycle_images, **images}
        self.losses = {**self.losses, **timecycle_losses}
        self.images = {**self.images, **timecycle_images}
        return timecycle_losses, timecycle_images


class TimeCycleMixinPaired(TimeCycleMixin, PairedModel):
    """
    Paired Timecycle
    Defines a conditional timecycle on self.fake_target, conditioned on self.real_source
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.timecycle_type == "conditional":
            self.define_timecycle(
                input_channels=self.source_channels + self.target_channels,
                output_channels=self.target_channels,
            )
        elif self.timecycle_type == "unconditional":
            self.define_timecycle(
                input_channels=self.target_channels * 2,
                output_channels=self.target_channels,
            )

    def forward(self):
        super().forward()
        ignore_first_n = 1
        self.timecycle_forward(
            sequence=self.fake_target,
            conditioning_sequence=self.real_source,
            ignore_first_n=ignore_first_n,
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_target,
            conditioning_sequence=self.real_source,
            ignore_first_n=ignore_first_n,
        )


class TimeCycleMixinUnpaired(TimeCycleMixin, UnpairedModel):
    """
    Unpaired Timecycle
    Defines two conditional timecycles on self.fake_target and self.fake_source,
    conditioned on self.real_source and self.real_target respectively
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.timecycle_type == "conditional":
            self.define_timecycle(
                input_channels=self.source_channels + self.target_channels,
                output_channels=self.target_channels,
                timecycle_name="target",
            )
            self.define_timecycle(
                input_channels=self.source_channels + self.target_channels,
                output_channels=self.source_channels,
                timecycle_name="source",
            )
        elif self.timecycle_type == "unconditional":
            self.define_timecycle(
                input_channels=self.target_channels * 2,
                output_channels=self.target_channels,
                timecycle_name="target",
            )
            self.define_timecycle(
                input_channels=self.source_channels * 2,
                output_channels=self.source_channels,
                timecycle_name="source",
            )
        # when using timerecycleganv2 we want to use unconditional timecycle, but we don't need the motion models
        elif self.timecycle_type == "timerecycle":
            self.timecycle_type = "unconditional"

    def forward(self):
        super().forward()
        ignore_first_n = 0
        self.timecycle_forward(
            sequence=self.fake_target,
            conditioning_sequence=self.real_source,
            timecycle_name="target",
            ignore_first_n=ignore_first_n,
        )
        self.timecycle_forward(
            sequence=self.fake_source,
            conditioning_sequence=self.real_target,
            timecycle_name="source",
            ignore_first_n=ignore_first_n,
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_target,
            timecycle_name="target",
            ignore_first_n=ignore_first_n,
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_source,
            timecycle_name="source",
            ignore_first_n=ignore_first_n,
        )

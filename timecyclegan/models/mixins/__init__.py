from .timecycle_mixin import (
    TimeCycleMixin,
    TimeCycleMixinPaired,
    TimeCycleMixinUnpaired
)
from .sequence_discriminator_mixin import (
    SeqDMixin,
    SeqDMixinPaired,
    SeqDMixinUnpaired
)
from .warp_loss_mixin import WarpLossMixin

# pylint: disable=abstract-method


class TimecycleSeqDMixinPaired(TimeCycleMixin, SeqDMixinPaired):
    """
    Paired Timecycle
    Defines a conditional timecycle on self.fake_target, conditioned on self.real_source
    Also defines a sequence discriminator that will be run on both generator fakes and timecycle fakes
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_timecycle(
            input_channels=self.source_channels + self.target_channels,
            output_channels=self.target_channels,
        )
        self.add_timecycle_discriminator(
            discriminator=self.discriminator_seq,
            n_frames=self.n_frames_D,
        )

    def forward(self):
        super().forward()
        self.timecycle_forward(
            sequence=self.fake_target,
            conditioning_sequence=self.real_source,
            ignore_first_n=self.n_frames_ignored
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_target,
            conditioning_sequence=self.real_source,
            ignore_first_n=self.n_frames_ignored,
        )


class TimecycleSeqDMixinUnpaired(TimeCycleMixin, SeqDMixinUnpaired):
    """
    Unpaired Timecycle + Sequence discriminator
    Defines two conditional timecycles on self.fake_target and self.fake_source,
    conditioned on self.real_source and self.real_target respectively
    Also defines two sequence discriminators that will be run on both generator fakes and timecycle fakes
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.add_timecycle_discriminator(
            discriminator=self.discriminator_target_seq,
            n_frames=self.n_frames_D,
            timecycle_name="target",
        )
        self.add_timecycle_discriminator(
            discriminator=self.discriminator_source_seq,
            n_frames=self.n_frames_D,
            timecycle_name="source",
        )

    def forward(self):
        super().forward()
        self.timecycle_forward(
            sequence=self.fake_target,
            conditioning_sequence=self.real_source,
            timecycle_name="target",
            ignore_first_n=self.n_frames_ignored,
        )
        self.timecycle_forward(
            sequence=self.fake_source,
            conditioning_sequence=self.real_target,
            timecycle_name="source",
            ignore_first_n=self.n_frames_ignored,
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_target,
            ignore_first_n=self.n_frames_ignored,
            timecycle_name="target",
        )
        self.set_timecycle_discriminator_input(
            real_sequence=self.real_source,
            ignore_first_n=self.n_frames_ignored,
            timecycle_name="source",
        )

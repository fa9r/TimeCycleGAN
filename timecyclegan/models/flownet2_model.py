"""A simple wrapper for flownet2, so we use our training/testing scripts with it"""

from .base_model import PairedModel
from .networks.flownet2 import FlowNet2, flow_to_img


class Flownet2Model(PairedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flownet = FlowNet2(self.device)

    def set_input(self, input_data):
        self.real_source = input_data['source'].to(self.device)

    def forward(self):
        pass

    def backward_D(self):
        pass

    def backward_G(self):
        pass

    def define_logging(self):
        pass

    def test(self):
        flow = self.flownet(self.real_source[:, self.source_channels:], self.real_source[:, :self.source_channels])[0]
        return flow_to_img(flow, height=self.height, width=self.width)

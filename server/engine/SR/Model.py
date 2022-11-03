import sys

import torch
import yaml
from basicsr.utils.options import ordered_yaml

from server.util.types import EngineDevice

sys.path.insert(0, "server/plugin/HAT")

from server.plugin.HAT.hat.models.hat_model import HATModel

MODEL_DEFINITION = "server/plugin/HAT/resource/HAT-L_SRx4_ImageNet-pretrain.yml"


class SRModel(HATModel):
    __device: EngineDevice

    def __init__(self, device: EngineDevice):
        self.__device = device

        with open(MODEL_DEFINITION, mode="r") as f:
            options = yaml.load(f, Loader=ordered_yaml()[0])

        options["is_train"] = False
        options["dist"] = False

        super().__init__(options, device=str(device))

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        self.lq = input.to(self.__device.torch())
        self.pre_process()
        self.process()
        self.post_process()
        return self.output

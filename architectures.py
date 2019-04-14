"""
This module defines the Architecture function which is a convenient way of defining
a desirable architecture (that is, creating a Network instance).
"""

from typing import Callable

import torch
import torch.nn as nn
import torchvision.models

import settings
import utils
from network import Network


class CNN(utils.AutoUniqueEnum):
    SqueezeNet = [lambda: torchvision.models.squeezenet1_1(pretrained=True)]
    ResNet34 = [lambda: torchvision.models.resnet34(pretrained=True)]

    def __init__(self, getInstance: Callable[[], nn.Module]):
        self.getInstance = getInstance[0]


class RNN(utils.AutoUniqueEnum):
    ElmanRNN_1l = lambda input_size, hidden_size: nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                                                         batch_first=True), False
    LSTM_1l = lambda input_size, hidden_size: nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                                                      batch_first=True), True
    LSTM_2l = lambda input_size, hidden_size: nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                                                      batch_first=True), True
    LSTM_3l = lambda input_size, hidden_size: nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3,
                                                      batch_first=True), True

    def __init__(self, getInstance: Callable[[int, int], nn.Module], twoStates: bool):
        self.getInstance = getInstance
        self._twoStates = twoStates

    @property
    def twoStates(self):
        return self._twoStates


class OutputNet(utils.AutoUniqueEnum):
    FC0l = [lambda input_size, output_size: utils.fullyConnected([input_size, output_size])]
    FC1l = [lambda input_size, output_size: utils.fullyConnected([input_size, input_size, output_size])]
    FC3l = [lambda input_size, output_size: utils.fullyConnected(
        [input_size, input_size, input_size, input_size, output_size])]

    def __init__(self, getInstance: Callable[[int, int], nn.Module]):
        self._getInstance = getInstance[0]

    @property
    def getInstance(self):
        return self._getInstance


def Architecture(cnn: CNN, rnn: RNN, outputNet: OutputNet, vocabLen, maxOutputLen, modelPath=None):
    """
    This method is a convenient way of defining a desirable architecture (that is, creating a Network instance).
    The returned model is stored in settings.device.
    """
    ret = Network(cnn.getInstance(), rnn.getInstance(vocabLen + 1, 1000), rnn.twoStates,
                  outputNet.getInstance(1000, vocabLen + 1), vocabLen, maxOutputLen).to(settings.device)
    if modelPath is None:
        return ret
    ret.load_state_dict(torch.load(modelPath))
    return ret

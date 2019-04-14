"""
This module configures the settings of system.
"""

from pathlib import Path

import torch

device = torch.device('cuda:0')

vocabfilepath = Path('vocab/vocab.vocab')
modelfilepath = Path('models/1/model1.pt')
visfilepath = Path('models/1/model1.jpg')
logname = 'model1'

printevery = 5000

valLen = 5000
samplesLen = 5


def sysAsserts():
    assert torch.backends.mkl.is_available()
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled

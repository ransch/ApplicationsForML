"""
This module contains all the hyperparameters of the model.
It contains a function for each module in the project (that uses at least one hyperparameter);
each function returns a named tuple with the parameters.
"""

from collections import namedtuple

_NetworkT = namedtuple('_NetworkT', ['topk'])
_COCODatasetT = namedtuple('_COCODatasetT',
                           ['FlipProb', 'RotationProb', 'ColorJitterProb', 'RotationDegrees', 'ColorJitterBrightness',
                            'ColorJitterContrast', 'ColorJitterSaturation', 'ColorJitterHue'])
_utilsT = namedtuple('_utilsT', ['FCReLUSlope'])
_mainT = namedtuple('_mainT',
                    ['workersNum', 'batchSize', 'maxCaptionLenDelta', 'adamLr', 'adamBetas', 'epochsNum', 'evalEvery'])

_Network = _NetworkT(25)
_COCODataset = _COCODatasetT(.5, .3, .3, 10, .1, .1, .1, .1)
_utils = _utilsT(.01)
_main = _mainT(0, 55, 5, .001, (0.9, 0.999), 3, 1)


def network():
    return _Network


def COCODataset():
    return _COCODataset


def utils():
    return _utils


def main():
    return _main

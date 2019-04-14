"""
This module defines utility methods that are used in the project.
"""

from enum import Enum

import torch
import torch.nn as nn

import settings
from hyperparameters import utils as Hyperparams


class Singleton(type):
    """
        This metaclass defines a Singleton. Each class that uses this metaclass follows the singleton design pattern.
    """

    _ins = {}

    def __call__(cls, *args, **kwargs):
        if cls in Singleton._ins:
            return Singleton._ins[cls]
        Singleton._ins[cls] = super().__call__(*args, **kwargs)
        return Singleton._ins[cls]


class AutoUniqueEnum(Enum):
    """
    This Enum is the base for enums that prohibit aliases and generates the values
    automatically. Subclass of this enum may assign custom attributes to the enumeration members.
    """

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, *args, **kwargs):
        self._value = self._value_

    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


def randSamp(vector, topk):
    """
    This method randomly picks a number in $\{ 0,...,len(vectors) - 1 \}$ by:
    1. Looking at the topk indices with the largest topk values
    2. Turning this topk-length vector into a probability distribution via softmax
    3. Randomly picking an index based on the probability distribution we have calculated
    """

    probs, indices = vector.topk(topk)
    probs = nn.functional.softmax(probs, dim=0)

    # Write $\{x_i\}_{i=0}^{len(probs)}, x_0 := 0, x_{i+1}:=x_i+probs[i]$
    # Thus, if $X \sim Unif(0,1)$, then $\mathbb{P} (X \in [x_i, x_{i+1}]) = probs[i]$
    s = torch.empty(1).uniform_(0, 1).item()

    # Now we should find the $i$ such that $s \in [x_i, x_{i+1})$
    xi = torch.zeros(1, dtype=torch.float32).to(settings.device)
    ind = len(probs) - 1
    for i in range(1, len(probs) + 1):
        xim1 = xi.item()
        xi.add_(probs[i - 1])
        if xim1 <= s <= xi.item():
            ind = i - 1
            break

    return indices[ind]


def oneHot(size, ind):
    """
    This method returns a one-hot vector.

    Parameters:
        size:
            The size of the (one dimensional) returned vector
        ind:
            The index of the only element of the vector that would be 1

    Returns:
        A tensor $ret$ such that
        $len(ret) = size \wedge \forall i \in \{0,...,size-1\}, ret[i]=\delta_{i,ind}$

    """

    ret = torch.zeros(size, dtype=torch.int8, requires_grad=False, device=settings.device)
    ret[ind] = 1
    return ret.detach()


def fullyConnected(sizes):
    """
    This method returns a simple fully connected feed forward network with len(sizes) - 1 layers.
    Each layer is initialized using the normal He initialization, the nonlinearity function used
    is leaky-ReLU.

    Parameters:
        sizes:
            For each 0 <= i < len(sizes) - 1, size[i] is the input size of the (one dimensional) (i+1)th layer;
            size[len(sizes) - 1] is the size of the output
    """

    modules = []
    for i in range(1, len(sizes) - 1):
        l = nn.Linear(sizes[i - 1], sizes[i])
        torch.nn.init.kaiming_normal_(l.weight, a=Hyperparams().FCReLUSlope, nonlinearity='leaky_relu')
        modules.append(l)
        modules.append(nn.LeakyReLU(negative_slope=Hyperparams().FCReLUSlope))
    modules.append(nn.Linear(sizes[len(sizes) - 2], sizes[len(sizes) - 1]))

    return nn.Sequential(*modules)


def indicesToSentence(vocab, indices):
    """
    This method gets a list of indices that form a caption, and returns the string that is represented by this list.

    Parameters:
        vocab:
            A tuple that represents the vocabulary
        indices:
            An array of indices that represents a caption

    Returns:
        The string that is represented by indices
    """
    terminateIndex = indices.index(len(vocab)) if len(vocab) in indices else len(indices)
    return ' '.join([vocab[i] for i in indices[:terminateIndex]]).capitalize()

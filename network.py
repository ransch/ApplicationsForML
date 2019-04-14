"""
This module defines the main neural network of the project.
"""

import torch
import torch.nn as nn

import settings
import utils
from hyperparameters import network as Hyperparams


class Network(nn.Module):
    """
    This class defines the main neural network of the project.
    It receives RGB images of shape (3*H*W) (H, W >= 224) whose entries values
    range from 0 to 1 and normalized using mean = [0.485, 0.456, 0.406] and
    std = [0.229, 0.224, 0.225], and outputs a sequence of words that form
    a natural-language description of the image. Each word in the sequence
    is represented by its index in self.vocab.
    """

    def __init__(self, cnn, rnn, twoStates, outputNet, vocabLen, maxOutputLen):
        """
        Initialize the network.

        Parameters:
            cnn:
                An instance of a convolutional neural network
                that would preprocess the input image and extract features from it

            rnn:
                An instance of a recurrent neural network
                that would output the image caption

            twoStates:
                A boolean that indicates whether rnn has two hidden states or one

            outputNet:
                An instance of a neural network that translates
                the hidden states of rnn into output words

            vocabLen:
                The length of the vocabulary

            maxOutputLen:
                The maximal allowed caption length

        Preconditions:
            - cnn should receive a tensor of shape (3*H*W) (H, W >= 224)
            whose entries values range from 0 to 1 and normalized using
            mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

            - The shape of the (one dimensional) tensor that cnn outputs should
            be equal to the shape of the hidden state of rnn

            - rnn expects to receive a one dimensional tensor with len(self.vocab)
              elements as an input

            - rnn has an integer attribute named hidden_size that equals the
            shape of the (one dimensional) hidden state tensor

            - rnn has an integer attribute named num_layers that equals the
            number of recurrent layers that are stacked together

            - rnn is batch-first
        """

        super().__init__()
        self._cnn = cnn
        self._rnn = rnn
        self._twoStates = twoStates
        self._outputNet = outputNet
        self._vocabLen = vocabLen
        self._maxOutputLen = maxOutputLen

    def forward(self, x):
        """
        This method defines the forward pass of the network.

        Parameters:
            x:
                The input image tensor

        Returns:
            - A list of indices that forms a caption to the given image -
            each element in the list is an index in vocab
            - A list of the probabilities for each word in the generated caption
        """
        batchSize = len(x)
        outputSeq = [[] for _ in range(batchSize)]

        inp = torch.zeros(batchSize, 1, self._rnn.input_size, requires_grad=False,
                          device=settings.device)
        inp_ = torch.zeros(batchSize, 1, self._rnn.input_size, requires_grad=False,
                           device=settings.device)

        first_hidden = self._cnn(x)

        hidden_elems = []
        for l in range(self._rnn.num_layers):
            hidden_elems.append(first_hidden)
        hidden = torch.stack(hidden_elems, 0)
        assert hidden.size() == (self._rnn.num_layers, batchSize, self._rnn.hidden_size)

        if self._twoStates:
            cell = torch.stack(hidden_elems, 0)
            assert hidden.size() == (self._rnn.num_layers, batchSize, self._rnn.hidden_size)

        outputLogProbs_elems = []
        for step in range(self._maxOutputLen + 1):  # plus 1 for the termination char
            if self._twoStates:
                _, (hidden, cell) = self._rnn(inp, (hidden, cell))
            else:
                _, hidden = self._rnn(inp, hidden)

            output = self._outputNet(hidden[-1])
            outputLogProbs_elems.append(nn.functional.log_softmax(output, dim=1))

            for b in range(batchSize):
                w = utils.randSamp(output[b], Hyperparams().topk)
                outputSeq[b].append(w)
                inp_[b, 0] = utils.oneHot(self._vocabLen + 1, w)

            inp = inp_.detach().add(0)

        outputLogProbs = torch.stack(outputLogProbs_elems, dim=1)
        assert outputLogProbs.size() == (batchSize, self._maxOutputLen + 1, self._vocabLen + 1)

        return outputSeq, outputLogProbs

import datetime
import pickle

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import architectures as archs
from COCODataset import COCODataset
from mainAux import *
from train import train

if __name__ == '__main__':
    settings.sysAsserts()
    filesAsserts()
    dataset = COCODataset('coco/images/train2017', 'coco/annotations/captions_train2017.json', True)
    with open(settings.vocabfilepath, 'rb') as dsf:
        pcklr = pickle.Unpickler(dsf)
        obj = pcklr.load()
        vocab = obj.get('vocab')
        maxCaptionLen = obj.get('maxCaptionLen')
        wordToInd = obj.get('wordToInd')

    trainLen = len(dataset) - settings.valLen
    inds = np.arange(len(dataset))
    np.random.shuffle(inds)

    cnn, rnn, outputNet = archs.CNN.ResNet34, archs.RNN.LSTM_2l, archs.OutputNet.FC1l
    model = archs.Architecture(cnn, rnn, outputNet, len(vocab), maxCaptionLen + Hyperparams().maxCaptionLenDelta)
    trainDl = DataLoader(dataset, batch_size=Hyperparams().batchSize, shuffle=False,
                         sampler=torch.utils.data.SubsetRandomSampler(inds[:trainLen]),
                         num_workers=Hyperparams().workersNum)
    valDl = DataLoader(dataset, batch_size=Hyperparams().batchSize, shuffle=False,
                       sampler=torch.utils.data.SubsetRandomSampler(inds[trainLen:]),
                       num_workers=Hyperparams().workersNum)
    optimizer = optim.Adam(model.parameters(), lr=Hyperparams().adamLr, betas=Hyperparams().adamBetas)

    printlog(str(datetime.datetime.now()))
    printlog(f'Training {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    printlog(f'The model: CNN = {cnn.name}, RNN = {rnn.name}, Output Net = {outputNet.name}\n')

    try:
        elapsed_time = train(model, trainDl, valDl, trainLen, settings.valLen, criterion, optimizer,
                             None, vocab, wordToInd, Hyperparams().epochsNum,
                             Hyperparams().evalEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback,
                             betterCallback, endCallback)

        elapsed_time = time.gmtime(elapsed_time)
        printlog(
            f'Training finished in {elapsed_time.tm_mday - 1} days, '
            f'{elapsed_time.tm_hour} hours, {elapsed_time.tm_min} minutes')
        plotLearningCurves()

    except:
        printlog('An error occurred :(')

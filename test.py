"""
This module defines the test function that tests a model.
"""

import pickle

import nltk.translate.bleu_score as bleu
import torch

import architectures as archs
import settings
import utils
from COCODataset import COCODataset
from hyperparameters import main as MainHyperparams
from mainAux import criterion


def test(modelPath):
    settings.sysAsserts()
    dataset = COCODataset('coco/images/val2017', 'coco/annotations/captions_val2017.json', True)
    with open(settings.vocabfilepath, 'rb') as dsf:
        pcklr = pickle.Unpickler(dsf)
        obj = pcklr.load()
        vocab = obj.get('vocab')
        maxCaptionLen = obj.get('maxCaptionLen')
        wordToInd = obj.get('wordToInd')

    cnn, rnn, outputNet = archs.CNN.ResNet34, archs.RNN.LSTM_2l, archs.OutputNet.FC1l
    model = archs.Architecture(cnn, rnn, outputNet, len(vocab),
                               maxCaptionLen + MainHyperparams().maxCaptionLenDelta, modelPath)

    loss = .0
    references = []
    hypotheses = []
    chencherry = bleu.SmoothingFunction()

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset.coco)):
            image, captions = dataset.coco[i]
            image = image.unsqueeze(0).to(settings.device)
            captions = [c.lower() for c in captions]
            outputSeqs, outputLogProbs = model(image)
            outputSeq = [utils.indicesToSentence(vocab, outputSeqs[0])]

            loss += criterion(outputLogProbs, outputSeq, wordToInd)
            references.append(captions)
            hypotheses.append(outputSeq)

            if i % 100 == 0:
                print(f'Evaluated {i + 1}/{len(dataset.coco)}')
                print(f'-----{(dataset.coco.coco.imgs[dataset.coco.ids[i]])["file_name"]}')
                print(f'Truth: {[c.capitalize() for c in captions]}')
                print(f'Prediction: {outputSeq[0].capitalize()}')
                print("\n")

    loss /= len(dataset.coco)
    bleu2 = bleu.corpus_bleu(references, hypotheses, weights=(0.5, 0.5),
                             smoothing_function=chencherry.method1)
    bleu3 = bleu.corpus_bleu(references, hypotheses, weights=(1. / 3., 1. / 3., 1. / 3.),
                             smoothing_function=chencherry.method1)
    bleu4 = bleu.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=chencherry.method1)

    print('--------------------------------------------------------------')
    print(f'Testing dataset consists of {len(dataset.coco)} samples')
    print(f'Negative log-likelihood loss = {loss.item()}')
    print(f'Perplexity = {2 ** loss.item()}')
    print(f'BLEU-2 score = {bleu2}')
    print(f'BLEU-3 score = {bleu3}')
    print(f'BLEU-4 score = {bleu4}')

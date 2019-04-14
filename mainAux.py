import time

import torch
from google.cloud import logging
from matplotlib import pyplot as plt

import settings
import utils
from COCODataset import splitCaption
from hyperparameters import main as Hyperparams


def filesAsserts():
    assert settings.vocabfilepath.is_file()
    assert not settings.modelfilepath.is_file()
    assert not settings.visfilepath.is_file()


_logging_client = logging.Client()
_logger = _logging_client.logger(settings.logname)
_trainLs = []
_valLs = []


def printlog(s):
    """
    This method prints a message to the output stream, and logs in to the cloud.

    Parameters:
        s:
            The message to print and log
    """

    print(s)
    _logger.log_text(s)


def epochCallback(epochs_num, epoch):
    printlog(f'Epoch {epoch} / {epochs_num}')


def progressCallback(so_far, remaining_time):
    remaining_time = time.gmtime(remaining_time)
    printlog(
        f'Processed {so_far} images; Remaining time: {remaining_time.tm_mday - 1} days, '
        f'{remaining_time.tm_hour} hours, {remaining_time.tm_min} minutes')


def evalEveryCallback():
    printlog(f'Starting model evaluation')


def lossCallback(training_loss, validation_loss):
    printlog(f'Training loss: {round(training_loss, 2)} , Validation loss: {round(validation_loss, 2)}')
    _trainLs.append(training_loss)
    _valLs.append(validation_loss)


def betterCallback(model, valDl, vocab):
    torch.save(model.state_dict(), settings.modelfilepath)
    printlog('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        model.eval()

        batch = next(iter(valDl))
        images, captions, fileNames = batch['image'].to(settings.device), batch['caption'], batch['fileName']
        assert len(images) >= settings.samplesLen

        for m in range(settings.samplesLen):
            image = images[m].unsqueeze(0)
            caption = captions[m]
            fileName = fileNames[m]
            outputSeqs, _ = model(image)
            outputSeq = utils.indicesToSentence(vocab, outputSeqs[0])
            printlog(f'-----{fileName}')
            printlog(f'-----Truth: {caption.capitalize()}')
            printlog(f'-----Predicted: {outputSeq}')


def endCallback():
    print('\n')


def plotLearningCurves():
    plt.plot(range(1, Hyperparams().epochsNum + 1, Hyperparams().evalEvery), _trainLs, '--o')
    plt.plot(range(1, Hyperparams().epochsNum + 1, Hyperparams().evalEvery), _valLs, '--o')
    plt.legend(('Training loss', 'Validation loss'))
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(str(settings.visfilepath), dpi=600)


def criterion(outputLogProbs, captions, wordToInd):
    loss = torch.zeros(1, device=settings.device)
    batchSize = len(captions)

    for b in range(batchSize):
        for t, word in enumerate(splitCaption(captions[b])):
            if word in wordToInd:
                ind = wordToInd[word]
                loss.add_(-1, outputLogProbs[b, t, ind])
            else:
                loss.add_(batchSize)
        loss.add_(-1, outputLogProbs[b, t + 1, outputLogProbs.size(2) - 1])

    loss.div_(batchSize)
    return loss

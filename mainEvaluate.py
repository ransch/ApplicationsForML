"""
This module is used for evaluating a model.
"""

import pickle

import torch
from PIL import Image
from torchvision import transforms

import architectures as archs
import settings
import utils
from hyperparameters import main as Hyperparams

_imagePrep = transforms.Compose(
    [
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

if __name__ == '__main__':
    with open(settings.vocabfilepath, 'rb') as dsf:
        pcklr = pickle.Unpickler(dsf)
        obj = pcklr.load()
        vocab = obj.get('vocab')
        maxCaptionLen = obj.get('maxCaptionLen')
        wordToInd = obj.get('wordToInd')

    cnn, rnn, outputNet = archs.CNN.ResNet34, archs.RNN.LSTM_2l, archs.OutputNet.FC1l
    model = archs.Architecture(cnn, rnn, outputNet, len(vocab), maxCaptionLen + Hyperparams().maxCaptionLenDelta,
                               'models/1/model1.pt')
    model.eval()

    path = input("Please enter the path of the image: ");

    with torch.no_grad():
        try:
            image = Image.open(path).convert('RGB')  # Converting for solving issues with RGBA images
            imageTensor = _imagePrep(image)
            image.close()
            imageTensor = imageTensor.unsqueeze(0).to(settings.device)
            outputSeqs, outputLogProbs = model(imageTensor)
            outputSeq = utils.indicesToSentence(vocab, outputSeqs[0])

            print("The predicted caption: ", outputSeq)
        except IOError:
            print("Cannot open the image specified")

# Applications For ML - Final Project
This repository contains the final project of the course "Applications For ML", which is an image caption generator.

The architecture follows the encoder-decoder scheme -
an input image is encoded via a CNN into a lower dimensional vector (that functions as a latent variable),
and then the encoded vector is used as a seed for the LSTM-based decoder which returns the output sequence.

A sample model trained on the COCO dataset may be found under /models directory.
Also, mainEvaluate.py may be used for running this model on a single image.

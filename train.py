"""
This module defines the train function that trains a model.
"""

import time

import math
import torch

import settings


def train(model, trainDl, valDl, trainLen, valLen, criterion, optimizer, scheduler, vocab, wordToInd, epochsNum,
          evalEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    best_validation_loss = math.inf
    sofar = 0
    printevery = settings.printevery

    model.train()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)
        if scheduler is not None:
            scheduler.step()

        for batch in trainDl:
            images, captions = batch['image'].to(settings.device), batch['caption']

            optimizer.zero_grad()
            outputSeqs, outputProbs = model(images)
            loss = criterion(outputProbs, captions, wordToInd)
            loss.backward()
            optimizer.step()

            sofar += len(images)
            if sofar >= printevery:
                progressCallback(sofar, (time.time() - start_time) / sofar * (
                        trainLen * epochsNum + (trainLen + valLen) * int((epochsNum + 1) / evalEvery) - sofar))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            model.eval()
            evalEveryCallback()
            training_loss, processed1 = totalLoss(model, trainDl, trainLen, criterion, wordToInd)
            validation_loss, processed2 = totalLoss(model, valDl, valLen, criterion, wordToInd)
            sofar += processed1 + processed2
            lossCallback(training_loss, validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                betterCallback(model, valDl, vocab)
            model.train()
        printevery = sofar
        endCallback()

    return time.time() - start_time


def totalLoss(model, dl, dlLen, criterion, wordToInd):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dl:
            images, captions = batch['image'].to(settings.device), batch['caption']

            processed += len(images)
            outputSeqs, outputProbs = model(images)
            loss += criterion(outputProbs, captions, wordToInd).item() * images.size(0)
        loss /= dlLen
    return loss, processed

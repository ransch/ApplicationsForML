"""
This script creates a COCODataset instance and serializes it into the filesystem.
"""

import pickle

from tqdm import tqdm

import settings
from COCODataset import COCODataset


def _wordToInd(vocab):
    ret = {}
    for i, word in tqdm(enumerate(vocab)):
        assert word not in ret
        ret[word] = i
    return ret


if __name__ == '__main__':
    assert not settings.vocabfilepath.is_file()

    with open(settings.vocabfilepath, 'wb') as dsf:
        pcklr = pickle.Pickler(dsf)
        dataset = COCODataset('coco/images/train2017', 'coco/annotations/captions_train2017.json', True)

        vocab = dataset.calcVocab()
        print('Vocabulary length:', len(vocab))

        maxCaptionLen = dataset.calcMaxCaptionLen()
        print('Max caption length:', maxCaptionLen)

        wordToInd = _wordToInd(vocab)

        obj = {'vocab': vocab, 'maxCaptionLen': maxCaptionLen, 'wordToInd': wordToInd}
        pcklr.dump(obj)

    print('Done!')

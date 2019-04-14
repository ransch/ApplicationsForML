"""
This module defines the dataset used in the project.
"""

import re

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from hyperparameters import COCODataset as Hyperparams


class _WithProb:
    """
    This class defines a transform that applies another one at a given probability.
    """

    def __init__(self, prob, transInstance):
        """
        Initializes the transform.

        Parameters:
            prob:
                The probability of applying the given transformation
            transInstance:
                An instance of some transformation

        Preconditions:
            - 0 <= prob <= 1

            - transInstance defines a __call__ method that receives an Image
            and returns an Image
        """

        self.prob = prob
        self.transInstance = transInstance

    def __call__(self, image):
        if torch.empty(1).bernoulli_(self.prob).item() == 1:
            return self.transInstance(image)
        else:
            return image


_imageAugment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(Hyperparams().FlipProb),
        _WithProb(Hyperparams().RotationProb, transforms.RandomRotation(Hyperparams().RotationDegrees)),
        _WithProb(Hyperparams().ColorJitterProb, transforms.ColorJitter(Hyperparams().ColorJitterBrightness,
                                                                        Hyperparams().ColorJitterContrast,
                                                                        Hyperparams().ColorJitterSaturation,
                                                                        Hyperparams().ColorJitterHue))])

_imagePrep = transforms.Compose(
    [
        transforms.Resize([300, 300]),
        transforms.ToTensor()
    ]
)


class COCODataset(Dataset):
    """
    This class defines the dataset used in the project.
    Note that as there are typically multiple captions that describe each image,
    on every query a random one is returned lowercased.
    """

    def __init__(self, images, annotations, augment, normalize=True):
        """
        Initializes the dataset.

        Parameters:
            images:
                The path of the directory that contains the images
            annotations:
                The path of the directory that contains the captions
            augment:
                Whether to use data augmentation techniques
            normalize:
                Whether to normalize the images
        """

        trans = transforms.Compose([_imageAugment, _imagePrep]) if augment else _imagePrep
        if normalize:
            trans = transforms.Compose(
                [trans, transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.coco = datasets.CocoCaptions(root=images, annFile=annotations,
                                          transform=trans)

    def __getitem__(self, index):
        image, captions = self.coco[index]
        randIndex = torch.empty(1, dtype=torch.uint8)
        randIndex.random_(0, len(captions))
        fileName = (self.coco.coco.imgs[self.coco.ids[index]])['file_name']
        return {'image': image, 'caption': captions[randIndex.item()].lower(), 'fileName': fileName}

    def __len__(self):
        return len(self.coco)

    def calcVocab(self):
        """
        This method calculates a sorted tuple of all the words
        in all the captions in the dataset.

        Preconditions:
            - self.coco should be already set

        Returns:
            A sorted tuple of all the words in all the captions in the dataset
        """
        ret = set()

        i = 0
        for _, captions in tqdm(self.coco):
            i += 1
            for caption in captions:
                ret.update(splitCaption(caption))

        return tuple(sorted(list(ret)))

    def calcMaxCaptionLen(self):
        """
        This method calculates the length of the longest caption in the dataset.

        Preconditions:
            - self.coco should be already set

        Returns:
            The length of the longest caption in the dataset
        """
        ret = 0

        for _, captions in tqdm(self.coco):
            for caption in captions:
                capLen = len(splitCaption(caption))
                if capLen > ret:
                    ret = capLen

        return ret


def splitCaption(caption):
    caption = caption.replace('\\', '').lower()
    return re.findall(r"[\w']+", caption)

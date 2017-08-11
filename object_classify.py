import os
import json
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision.transforms import Normalize
from PIL import Image
import numpy as np
import requests

squeezenet = models.squeezenet1_0(pretrained=True)

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

def process_labels(download=False):
    if download:
        labels = requests.get(LABELS_URL).json()
    else:
        with open('labels.json') as fp:
            labels = json.load(fp)
    return {int(key): value for key, value in labels.items()}

LABELS = process_labels()

def softmax_to_label(softmax, to_name=True):
    assert isinstance(softmax, Variable)
    _, argmax = softmax.max(1)
    label = int(np.asscalar(argmax.data.numpy()))
    if to_name:
        return LABELS[label]
    else:
        return label


normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])


def predict(img):
    if isinstance(img, str):
        img = Image.open(img)
    assert isinstance(img, Image.Image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
    img = img[None] # expand a fake batch dim
    img = img.astype(np.float32) / 255.
    img = normalize(torch.FloatTensor(img))
    prediction = squeezenet(Variable(img, volatile=True))
    return softmax_to_label(prediction)


if __name__ == '__main__':
    for img in os.listdir('demo_imgs'):
        img = os.path.join('demo_imgs', img)
        print('file {} -> {}'.format( img, predict(img) ))

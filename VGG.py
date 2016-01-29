#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class VGG(chainer.Chain):

    def __init__(self):
        super(VGG, self).__init__()
        links = [('conv1_1', L.Convolution2D(3, 64, 3, stride=1, pad=1))]
        links += [('conv1_2', L.Convolution2D(64, 64, 3, stride=1, pad=1))]
        links += [('_mpool1', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv2_1', L.Convolution2D(64, 128, 3, stride=1, pad=1))]
        links += [('conv2_2', L.Convolution2D(128, 128, 3, stride=1, pad=1))]
        links += [('_mpool2', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv3_1', L.Convolution2D(128, 256, 3, stride=1, pad=1))]
        links += [('conv3_2', L.Convolution2D(256, 256, 3, stride=1, pad=1))]
        links += [('conv3_3', L.Convolution2D(256, 256, 3, stride=1, pad=1))]
        links += [('_mpool3', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv4_1', L.Convolution2D(256, 512, 3, stride=1, pad=1))]
        links += [('conv4_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('conv4_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('_mpool4', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv5_1', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('conv5_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('conv5_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('_mpool5', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('fc6', L.Linear(25088, 4096))]
        links += [('_dropout6', F.Dropout(0.5))]
        links += [('fc7', L.Linear(4096, 4096))]
        links += [('_dropout7', F.Dropout(0.5))]
        links += [('fc8', L.Linear(4096, 1000))]

        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)

        self.forward = links

    def __call__(self, x, target_layer):
        for name, f in self.forward:
            x = f(x)
            if name == target_layer:
                break

        return x

""" LeNet Tensorflow 2.7.x implementation
@author Wajih Ouertani
@email wajih.ouertani@gmail.com
"""

from re import L
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dense,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Optimizer


def softmax(logits):
    """A helper that computes softmax"""
    exp_logits = np.exp(logits)
    if len(exp_logits.shape) > 1:
        # multiple image support
        return (
            exp_logits
            / np.tile(np.sum(exp_logits, axis=1), (exp_logits.shape[1], 1)).transpose()
        )
    else:
        # one image
        return exp_logits / np.sum(exp_logits)


def load_layers(layers: Optional[list] = None):
    if layers:
        return [layer for layer in layers]
    return []


class LeNet:

    """Tensorflow 2.x implementation of leNet architecture"""

    def __init_layers(self):
        """Initialize (core) layers

        The LeNet architecture accepts a 32x32xC image as input,
        where C is the number of color channels.

            Architecture:
                Conv. layers:
                    - Conv. layer 1: The output shape should be 28x28x6. + (default: ReLU activation)
                    - Pooling 1: The output shape should be 14x14x6.

                    - Conv. Layer 2: The output shape should be 10x10x16.  + (default: ReLU activation)
                    - Pooling 2: The output shape should be 5x5x16.

                    - Flatten: Flatten the output shape of the final pooling layer

                Fully connected layers:
                    - Layer 4: Fully Connected (default: Relu activation)
                    - Layer 3: Fully Connected (default: Relu activation)
                    - Layer 5: Fully Connected (Logits)

        """
        self.layers = [
            Conv2D(
                6, kernel_size=(5, 5), padding="valid", activation=self.conv_activation
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(
                16, kernel_size=(5, 5), padding="valid", activation=self.conv_activation
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dropout(0.3),
            Dense(120, activation=self.fcn_activation),
            Dense(84, activation=self.fcn_activation),
            Dense(self.categ_nbr)
            if self.logits
            else Dense(self.categ_nbr, activation="softmax"),
        ]

    def __init__(
        self,
        categ_nbr: int = 10,
        conv_activation: str = "relu",
        fcn_activation: str = "relu",
        logits=True,
        preprocessing_layers: Optional[list] = None,
        augmentation_layers: Optional[list] = None,
        name="UNNAMED",
    ):
        """Initialize network parameters"""
        self.categ_nbr = categ_nbr
        self.conv_activation = conv_activation
        self.fcn_activation = fcn_activation
        self.logits = logits

        self.preprocessing_layers = load_layers(preprocessing_layers)
        self.augmentation_layers = load_layers(augmentation_layers)

        self.__init_layers()

        self._train_model = None
        self._pred_model = None
        self._name = name

    @property
    def model(self):
        return self._train_model

    @property
    def pred_model(self):
        return self._pred_model

    @staticmethod
    def output(input_: Input, layers: Optional[List] = None):
        """a helper method to build model based given layers"""
        output_ = input_
        for layer in layers:
            output_ = layer(output_)
        return output_

    def build_models(self, input_: Input):
        """Initialize model(s)
        first (self._train_model) for training including data augmentation layers
        second (self._pred_model) for prediction ignoring the augmentation layers
        """
        self._train_model = Model(
            input_,
            self.output(
                input_,
                self.preprocessing_layers + self.augmentation_layers + self.layers,
            ),
            name=f"{self._name}.training",
        )

        # building prediction model
        self._pred_model = Model(
            input_,
            self.output(
                input_,
                self.preprocessing_layers
                + list(
                    filter(
                        lambda layer: layer.__class__.__name__ != "Dropout", self.layers
                    )
                ),
            ),
            name=f"{self._name}.prediction",
        )

    def _loss(self):
        """A helper to build CategoricalCrossentropy loss"""
        return CategoricalCrossentropy(
            from_logits=self.logits,
            name="categorical_crossentropy",
        )

    def compile_model(self, optimizer: Optimizer, metrics: List[str] = ["accuracy"]):
        """Compile (training) model"""
        self.model.compile(loss=self._loss(), optimizer=optimizer, metrics=metrics)

    def predict(self, images: np.array):
        shape = images.shape
        if len(shape) not in {3, 4}:
            raise ValueError("dimension error !")
        assert shape[-1] == 3
        if len(shape) == 3:
            return self.pred_model(np.expand_dims(images, axis=0))
        return self.pred_model.predict(images)

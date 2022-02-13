""" LeNet Tensorflow 2.7.x implementation
@author Wajih Ouertani
@email wajih.ouertani@gmail.com
"""

from re import L
from typing import List, Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from cv2 import batchDistance
from pydantic import BaseModel
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


def load_layers(layers: Optional[list] = None):
    if layers:
        return [layer for layer in layers]
    return []


class ConvLayerConfig(BaseModel):
    """A layer filter config (lite) a helper class to configure a layer filter."""

    filters: int
    kernel_size: Tuple[int, int] = (5, 5)  # default kernel size set to (5, 5)


class LeNet:

    """Tensorflow 2.x implementation of leNet architecture"""

    def __init_layers(self):
        """Initialize (core) layers

        The LeNet architecture accepts a 32x32xC image as input,
        where C is the number of color channels.

            Architecture:
                Conv. layers:
                    - Conv. layer 1: The output shape should be 28x28x6. + (default: ReLU activation)
                    - (optional batch normalization not orig.)
                    - Pooling 1: The output shape should be 14x14x6.

                    - Conv. Layer 2: The output shape should be 10x10x16.  + (default: ReLU activation)
                    - (optional batch normalization not orig.)
                    - Pooling 2: The output shape should be 5x5x16.

                    - Flatten: Flatten the output shape of the final pooling layer
                    - (optional dropout - not orig.)
                Fully connected layers:
                    - Layer 4: Fully Connected (default: Relu activation)
                    - Layer 3: Fully Connected (default: Relu activation)
                    - Layer 5: Fully Connected (Logits)

        """
        self.layers = [
            Conv2D(
                **self.conv_layers_config.get(
                    1, ConvLayerConfig(filters=6, kernel_size=(5, 5))
                ).dict(),
                padding="valid",
                activation=self.conv_activation,
            )
        ]
        if self._batch_norm:
            self.layers.append(BatchNormalization())

        self.layers.extend(
            [
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(
                    **self.conv_layers_config.get(
                        2, ConvLayerConfig(filters=16, kernel_size=(5, 5))
                    ).dict(),
                    padding="valid",
                    activation=self.conv_activation,
                ),
            ]
        )

        if self._batch_norm:
            self.layers.append(BatchNormalization())
        self.layers.extend([MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), Flatten()])
        if self._dropout:
            self.layers.append(Dropout(self._dropout))
        self.layers.extend(
            [
                Dense(120, activation=self.fcn_activation),
                Dense(84, activation=self.fcn_activation),
                Dense(self.categ_nbr)
                if self.logits
                else Dense(self.categ_nbr, activation="softmax"),
            ]
        )

    def __init__(
        self,
        categ_nbr: int = 10,
        conv_activation: str = "relu",
        fcn_activation: str = "relu",
        logits=True,
        preprocessing_layers: Optional[list] = None,
        augmentation_layers: Optional[list] = None,
        conv_layers_config: Optional[Dict[int, ConvLayerConfig]] = None,
        name="UNNAMED",
        batch_norm=True,
        dropout: float = 0.2,
    ):
        """Initialize network parameters"""
        self.categ_nbr = categ_nbr
        self.conv_activation = conv_activation
        self.fcn_activation = fcn_activation
        self.logits = logits

        self.preprocessing_layers = load_layers(preprocessing_layers)
        self.augmentation_layers = load_layers(augmentation_layers)

        #  default convolution layers config (we have 2 layers)
        self.conv_layers_config: Dict[int, ConvLayerConfig] = {
            1: ConvLayerConfig(filters=6, kernel_size=(5, 5)),
            2: ConvLayerConfig(filters=16, kernel_size=(5, 5)),
        }
        if conv_layers_config is not None:
            self.conv_layers_config.update(conv_layers_config)

        self._batch_norm = batch_norm
        self._dropout = dropout

        self.__init_layers()

        self._train_model = None
        self._pred_model = None
        self._name = name

    @property
    def name(self):
        return self._name

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

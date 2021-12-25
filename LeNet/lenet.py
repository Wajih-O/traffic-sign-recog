from typing import List, Optional
import tensorflow as tf

from tensorflow.keras.layers import (
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dense,
    BatchNormalization,
)


def LeNet(
    x,
    categ_nbr=10,
    conv_activation: str = "relu",
    fcn_activation: str = "relu",
    logit=True,
):
    """Tensorflow 2.1 + implementation  of leNet architecture

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

    layers = [
        Conv2D(6, kernel_size=(5, 5), padding="valid", activation=conv_activation),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), padding="valid", activation=conv_activation),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(120, activation=fcn_activation),
        Dense(84, activation=fcn_activation),
        Dense(categ_nbr) if logit else Dense(categ_nbr, activation="softmax"),
    ]
    x_output = x
    for layer in layers:
        x_output = layer(x_output)
    return x_output


def LeNetExp(
    x,
    categ_nbr=10,
    conv_activation: str = "relu",
    fcn_activation: str = "relu",
    logit=True,
    augmentation: Optional[List] = None,
):
    """Tensorflow 2.1 + implementation  of leNet architecture with batch normalization

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

                TODO: add an optional softmax application at the end. (now done externally)
    """
    layers = []
    if augmentation:
        layers += augmentation
    layers += [
        Conv2D(32, kernel_size=(5, 5), padding="valid", activation=conv_activation),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(5, 5), padding="valid", activation=conv_activation),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(120, activation=fcn_activation),
        Dense(84, activation=fcn_activation),
        Dense(categ_nbr) if logit else Dense(categ_nbr, activation="softmax"),
    ]
    x_output = x
    for layer in layers:
        x_output = layer(x_output)
    return x_output

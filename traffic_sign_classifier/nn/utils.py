import numpy as np


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

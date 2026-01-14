"""
Differentiable layers for genomics machine learning.

"""

from genoml.layers.conv import (
    Conv1D,
    MaxPool1D,
    GlobalMaxPool1D,
    GlobalAvgPool1D,
)

from genoml.layers.pwm import (
    PWMScanner,
    MotifLayer,
    create_pwm_from_sequences,
)

from genoml.layers.attention import (
    SelfAttention,
    PositionalEncoding,
    MultiHeadAttention,
)

from genoml.layers.dense import (
    Dense,
    BatchNorm1D,
    Dropout,
    LayerNorm,
)

__all__ = [
    # Convolutions
    "Conv1D",
    "MaxPool1D",
    "GlobalMaxPool1D",
    "GlobalAvgPool1D",
    # PWM
    "PWMScanner",
    "MotifLayer",
    "create_pwm_from_sequences",
    # Attention
    "SelfAttention",
    "PositionalEncoding",
    "MultiHeadAttention",
    # Dense
    "Dense",
    "BatchNorm1D",
    "Dropout",
    "LayerNorm",
]

from dataclasses import dataclass

import numpy as np


@dataclass
class EncodeRequest:
    input_img: np.ndarray

@dataclass
class EncodeResponse:
    latent: np.ndarray

@dataclass
class DecodeRequest:
    latent: np.ndarray

@dataclass
class DecodeResponse:
    output_img: np.ndarray

@dataclass
class Shutdown:
    pass
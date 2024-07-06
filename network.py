import torch.nn as nn

KERNEL_WH = 3
PADDING = 1

I_WH = 128

INPUT_CHANNELS = 3

CONV_LAYER_1_CHANNELS = 8
CONV_LAYER_2_CHANNELS = 16
CONV_LAYER_3_CHANNELS = 32
CONV_LAYER_4_CHANNELS = 64

NUM_POOLS = 3

LIN_LAYER_1_SIZE = 2048
LIN_LAYER_2_SIZE = 1024
LIN_LAYER_3_SIZE = 512
LIN_LAYER_4_SIZE = 256
LIN_LAYER_5_SIZE = 128

LATENT_SPACE = 64


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CONV_LAYER_1_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(CONV_LAYER_1_CHANNELS, CONV_LAYER_2_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(CONV_LAYER_2_CHANNELS, CONV_LAYER_3_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(CONV_LAYER_3_CHANNELS, CONV_LAYER_4_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(CONV_LAYER_4_CHANNELS * (I_WH // (2 ** NUM_POOLS)) * (I_WH // (2 ** NUM_POOLS)), LIN_LAYER_1_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_1_SIZE, LIN_LAYER_2_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_2_SIZE, LIN_LAYER_3_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_3_SIZE, LIN_LAYER_4_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_4_SIZE, LIN_LAYER_5_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_5_SIZE, LATENT_SPACE),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SPACE, LIN_LAYER_5_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_5_SIZE, LIN_LAYER_4_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_4_SIZE, LIN_LAYER_3_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_3_SIZE, LIN_LAYER_2_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_2_SIZE, LIN_LAYER_1_SIZE),
            nn.ReLU(True),

            nn.Linear(LIN_LAYER_1_SIZE, CONV_LAYER_4_CHANNELS * (I_WH // (2 ** NUM_POOLS)) * (I_WH // (2 ** NUM_POOLS))),
            nn.ReLU(True),

            nn.Unflatten(1, (CONV_LAYER_4_CHANNELS, (I_WH // (2 ** NUM_POOLS)), (I_WH // (2 ** NUM_POOLS)))),

            nn.ConvTranspose2d(CONV_LAYER_4_CHANNELS, CONV_LAYER_3_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.ConvTranspose2d(CONV_LAYER_3_CHANNELS, CONV_LAYER_2_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.ConvTranspose2d(CONV_LAYER_2_CHANNELS, CONV_LAYER_1_CHANNELS, KERNEL_WH, stride=1,
                               padding=PADDING),
            nn.ReLU(True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.ConvTranspose2d(CONV_LAYER_1_CHANNELS, INPUT_CHANNELS, KERNEL_WH, stride=1, padding=PADDING),
            nn.Sigmoid()
        )
        pass

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

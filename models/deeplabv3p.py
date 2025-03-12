import segmentation_models_pytorch as smp

class DeepLabV3Plus:
    def __init__(self, encoder_name='resnet34', in_channels=3, out_channels=1, activation=None):
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)
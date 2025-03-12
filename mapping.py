from models import UNet, UnetPlusPlus, DeepLabV3Plus
from metrics import DiceMetric
from loss import DiceLoss
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient, MeanSquaredError

# Models
model_mapping = {
    "unet": UNet,
    "resunetpp": UnetPlusPlus,
    "deeplab": DeepLabV3Plus
}

# Metrics
# metrics_mapping = {
#     "dsc": DiceMetric,
#     "loss": Loss(DiceLoss())
# }

# Loss
loss_mapping = {
    "loss": DiceLoss
}

# Monitor
mapping = {
    "lesion": {
        "model": model_mapping,
        # "metrics": metrics_mapping,
        "loss": loss_mapping
    }
}
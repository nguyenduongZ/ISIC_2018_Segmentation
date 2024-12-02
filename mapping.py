from torch import nn
from models import Unet
from metrics import miou, pixel_accuracy

# Model
enc_dec_mapping = {
    'unet' : Unet
}

# LOSSES
seg_loss = {
    "cross_entropy" : nn.CrossEntropyLoss()
}

# METRICS
seg_metrics_mapping = {
    'iou' : miou
}

# MONITOR
mapping = {
    'isic2018' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'loss' : seg_loss,
            'metrics' : seg_metrics_mapping
        },
    },
}
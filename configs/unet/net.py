import logging

from cvpods.layers import ShapeSpec
from unet import UNet




def build_model(cfg):

    model = UNet(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model

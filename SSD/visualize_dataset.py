import numpy as np
import os
import matplotlib.pyplot as plt
from train import get_parser
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from vizer.draw import draw_boxes
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils import box_utils


args = get_parser().parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

data_loader = make_data_loader(cfg, is_train=True)

mean = np.array([cfg.INPUT.PIXEL_MEAN]).reshape(1, 1, -1)
std = np.array([cfg.INPUT.PIXEL_STD])
priors = PriorBox(cfg)()
if isinstance(data_loader, list):
    data_loader = data_loader[0]
try:
    os.mkdir("transform_test")
except OSError:
    print ("directory already exists")
for idx, (img, batch, *_) in enumerate(data_loader):
    boxes = batch["boxes"]
    # SSD Target transform transfers target boxes into prior locations
    # Have to revert the transformation
    boxes = box_utils.convert_locations_to_boxes(
        boxes, priors, cfg.MODEL.CENTER_VARIANCE,
        cfg.MODEL.SIZE_VARIANCE
    )
    boxes = box_utils.center_form_to_corner_form(boxes)

    # Remove all priors that are background
    boxes = boxes[0]
    labels = batch["labels"][0].squeeze().cpu().numpy()
    boxes = boxes[labels != 0]
    labels = labels[labels != 0]
    # Resize to image widht and height
    boxes[:, [0, 2]] *= img.shape[3]
    boxes[:, [1, 3]] *= img.shape[2]
    img = img.numpy()
    # NCHW to HWC (only select first element of batch)
    img = np.moveaxis(img, 1, -1)[0]
    # Remove normalization
    img = img * std + mean

    img = np.clip(img.astype(np.uint8), a_min = 0, a_max = 255)
    img = draw_boxes(img, boxes, labels)
    plt.imsave("transform_test/img_%d.png" % idx, img)
    plt.clf()
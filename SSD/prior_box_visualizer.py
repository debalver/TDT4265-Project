import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.config.defaults import cfg
from train import get_parser
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.cm import jet


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    prior_box = PriorBox(cfg)
    prior_boxes = prior_box()

    feature_maps = cfg.MODEL.PRIORS.FEATURE_MAPS
    aspect_ratios = cfg.MODEL.PRIORS.ASPECT_RATIOS

    fig, axs = plt.subplots(len(feature_maps), figsize=(4, 3*len(feature_maps)))

    counter = 0
    for k, f in enumerate(feature_maps):
        num_boxes = 2 + 2 * len(aspect_ratios[k])
        prior_idx = f[0] * f[1] * num_boxes
        counter += prior_idx
        print(f)
        rectangles = []
        for prior_box in prior_boxes[counter-num_boxes:counter]:
            # prior_box of shape [center_x, center_y, w, h]
            rec = Rectangle((prior_box[0] - prior_box[2]/2, prior_box[1] - prior_box[3]/2), 
                            prior_box[2], prior_box[3])
            center = Circle((prior_box[0], prior_box[1]), radius=0.001)
            rectangles.append(rec)
            rectangles.append(center)
        np.random.seed(3)
        colors = [jet(x) for x in np.random.rand(num_boxes*2)]
        pc = PatchCollection(rectangles, alpha=0.5, lw=2, facecolor='None')
        pc.set_edgecolor(colors)
        axs[k].add_collection(pc)
        axs[k].set_title("%s - %d boxes" %(str(f), num_boxes))
        axs[k].autoscale()
    plt.tight_layout()
    plt.savefig('prior_boxes.png')

if __name__ == '__main__':
    main()
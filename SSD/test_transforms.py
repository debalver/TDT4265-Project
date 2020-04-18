import argparse
import os
import torch
from ssd.data.build import make_data_loader
from ssd.config.defaults import cfg
from torchvision.utils import save_image




def get_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    dataloader = make_data_loader(cfg, is_train=True, max_iter=10, start_iter=0)
    try:
        os.mkdir("transform_test")
    except OSError:
        print ("directory already exists")
    mean = torch.FloatTensor(cfg.INPUT.PIXEL_MEAN)
    for index, (images, targets, _) in enumerate(dataloader):
        for i in range(images.shape[0]):
            img = images[i] #torch.Size([3, 240, 320])
            # undo normalization:
            img = img.permute(1, 2, 0).float() + mean
            save_image(img.permute(2, 0, 1), "transform_test/img_%d_%d.png" % (index, i))


if __name__ == '__main__':
    main()

import argparse
import pprint
from collections import OrderedDict


class TrainOptions:

    def __init__(self):

        parser = argparse.ArgumentParser()

        # ------------------
        # dataset parameters
        # ------------------
        parser.add_argument('--image_root', type=str,
                            default='')
        parser.add_argument('--mask_root', type=str,
                            default='')

        parser.add_argument('--save_dir', type=str, default='snapshots/ckpt_celeba')
        parser.add_argument('--log_dir', type=str, default='runs/log_celeba')
        parser.add_argument('--pre_trained', type=str, default='')

        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=6)
        parser.add_argument('--load_size', type=int, default=(256, 256))
        parser.add_argument('--sigma', type=float, default=2.)
        parser.add_argument('--mode', type=str, default='train')

        parser.add_argument('--gen_lr', type=float, default=0.0002)
        parser.add_argument('--D2G_lr', type=float, default=0.1)
        parser.add_argument('--lr_finetune', type=float, default=0.00005)
        parser.add_argument('--finetune', type=bool, default=False)
        
        parser.add_argument('--start_iter', type=int, default=1)
        parser.add_argument('--train_iter', type=int, default=300000)
        parser.add_argument('--save_interval', type=int, default=10000)

        parser.add_argument('--VALID_LOSS', type=float, default=10.0)
        parser.add_argument('--HOLE_LOSS', type=float, default=60.0)
        parser.add_argument('--PERCEPTUAL_LOSS', type=float, default=0.1)
        parser.add_argument('--STYLE_LOSS', type=float, default=250.)
        parser.add_argument('--ADVERSARIAL_LOSS', type=float, default=0.1)
        parser.add_argument('--INTERMEDIATE_LOSS', type=float, default=1.0)

        # -----------
        # Distributed
        # -----------
        parser.add_argument('--local_rank', type=int, default=0, help="local rank for distributed training")

        self.opts = parser.parse_args()

    @property
    def parse(self):

        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts

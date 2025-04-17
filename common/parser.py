"""Argument parser functions."""

import argparse

from common.default_config import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    """
    parser = argparse.ArgumentParser(
        description="Tafar training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/NTU60/Non-Auto1.yaml",
    )
    parser.add_argument(
        "--opts",
        help="See common/default_config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg

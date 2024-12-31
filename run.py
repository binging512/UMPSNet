import os
from shutil import copyfile
import argparse
import yaml
from yacs.config import CfgNode
from train import train
from test import test
from utils.logger import Logger

def run(args):
    cfg = yaml.load(open(args.config,'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)

    os.makedirs(cfg.workspace,exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_val), exist_ok=True)
    if args.train_pass == True:
        copyfile(args.config, os.path.join(cfg.workspace, os.path.basename(args.config)))
    logger =Logger(cfg)
    logger.info(cfg)
    if args.train_pass == True:
        logger.info("Starting training pass....")
        train(cfg, logger)
    if args.test_pass == True:
        logger.info("Starting testing pass....")
        test(cfg, logger)
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--config', default='configs/default.yaml', type=str)
    parser.add_argument("--train_pass", action='store_true', default=False)
    parser.add_argument("--test_pass", action='store_true', default=False)
    args = parser.parse_args()
    run(args)
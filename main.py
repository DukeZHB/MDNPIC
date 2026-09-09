"""
MDNPIC: Multi-Granularity Conditional Guidance Diffusion Network with
Semantic Alignment Regularization for Pathology Image Classification.

Usage:
  python main.py --config configs/chaoyang.yml --doc chaoyang
  python main.py --config configs/chaoyang.yml --doc chaoyang --test --eval_best
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import yaml

parser = argparse.ArgumentParser(description=globals()["__doc__"])

parser.add_argument(
    "--config", type=str, required=True, help="Path to the config file")
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument(
    "--exp", type=str, default="exp", help="Path for saving running related data.")
parser.add_argument(
    "--doc", type=str, required=True,
    help="A string for documentation purpose. Will be the name of the log folder.")
parser.add_argument(
    "--dataroot", type=str, default=None,
    help="This argument will overwrite the dataroot in the config if it is not None.")
parser.add_argument(
    "--comment", type=str, default="", help="A string for experiment comment")
parser.add_argument(
    "--verbose", type=str, default="info",
    help="Verbose level: info | debug | warning | critical")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
parser.add_argument(
    "--resume_training", action="store_true", help="Whether to resume training")
parser.add_argument(
    "--train_guidance_only", action="store_true",
    help="Whether to only pre-train the MGCG guidance network")
parser.add_argument(
    "--noise_prior", action="store_true",
    help="Whether to apply a zero-mean noise prior distribution at timestep T")
parser.add_argument(
    "--add_ce_loss", action="store_true",
    help="Whether to add cross entropy as an auxiliary loss")
parser.add_argument(
    "--eval_best", action="store_true",
    help="Evaluate best model during training, instead of the ckpt stored at the last epoch")
parser.add_argument(
    "--n_splits", type=int, default=10,
    help="total number of runs with different seeds for a specific task")
parser.add_argument("--split", type=int, default=0, help="split ID")
parser.add_argument(
    "--ni", action="store_true",
    help="No interaction. Suitable for Slurm job launchers")
parser.add_argument(
    "--loss", type=str, default='mdnpic', help="loss function")
parser.add_argument(
    "--timesteps", type=int, default=None, help="number of diffusion steps")
parser.add_argument(
    "--num_sample", type=int, default=1,
    help="number of samples used in forward and reverse")

args = parser.parse_args()


def parse_config():
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join(args.config), encoding='utf-8') as f:
        if args.test:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    # overwrite if dataroot is not None
    if not args.dataroot is None:
        new_config.data.dataroot = args.dataroot
        new_config.data.traindata = args.dataroot
        new_config.data.testdata = args.dataroot

    if not args.test:
        args.im_path = os.path.join(args.exp, new_config.training.image_folder, args.doc)
        new_config.diffusion.noise_prior = True if args.noise_prior else False
        if not args.resume_training:
            if not args.timesteps is None:
                new_config.diffusion.timesteps = args.timesteps
            if args.num_sample > 1:
                new_config.diffusion.num_sample = args.num_sample
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
            if not os.path.exists(args.im_path):
                os.makedirs(args.im_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        import torch.utils.tensorboard as tb
        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        args.im_path = os.path.join(args.exp, new_config.testing.image_folder, args.doc)
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "testmetrics.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        os.makedirs(args.im_path, exist_ok=True)

    # add device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    config, logger = parse_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    if args.loss == 'mdnpic':
        from trainer import Diffusion
    else:
        raise NotImplementedError("Invalid loss option")

    try:
        runner = Diffusion(args, config, device=config.device)
        start_time = time.time()
        if args.test:
            runner.test()
            procedure = "Testing"
        else:
            runner.train()
            procedure = "Training"
        end_time = time.time()
        logging.info("\n{} procedure finished. It took {:.4f} minutes.\n\n\n".format(
            procedure, (end_time - start_time) / 60))
        # remove logging handlers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    args.doc = args.doc + "/split_" + str(args.split)
    if args.test:
        args.config = os.path.join(args.exp, "logs", args.config, "config.yml") \
            if not args.config.endswith(".yml") else args.config
    sys.exit(main())

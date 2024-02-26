import logging
import argparse

from utils.transform import get_transforms
from hnet import HyperNetwork

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
from mnet import VitNet
import pandas as pd

from utils.regularizer import Regularizer

from avalanche.benchmarks.classic import (
    SplitCIFAR10,
    SplitCIFAR100,
    SplitTinyImageNet,
)

torch.random.manual_seed(321)
np.random.seed(321)


def load_model(args):
    mnet = VitNet(args.blocks_fine_tuning, args.class_nums, args.peft)
    hnet = HyperNetwork(
        mnet.param_shapes,
        task_nums=args.task_nums,
        task_emb_dims=args.task_emb_size,
        block_emb_dims=args.block_emb_size,
        use_block_emb=args.use_block_emb,
        block_emb_sharing=args.block_emb_sharing,
        peft=args.peft,
        rank=args.rank,
        prefix_length=args.prefix_length,
        down_sample_dims=args.down_sample_dims
    )
    return hnet, mnet


def load_scenario(args):
    assert args.dataset in [
        "splitmnist",
        "rotatedmnist",
        "permutedmnist",
        "splitcifar10",
        "splitcifar100",
        "splittinyimagenet",
        "splitimagenet"
    ]

    if args.dataset == "splitcifar10":
        train_transforms, test_transforms = get_transforms(args)
        scenario = SplitCIFAR10(
            n_experiences=args.task_nums,
            return_task_id=True,
            dataset_root="./data/cifar10",
            class_ids_from_zero_in_each_exp=True,
            class_ids_from_zero_from_first_exp=False,
            train_transform=train_transforms,
            eval_transform=test_transforms,
        )

        return scenario
    elif args.dataset == "splitcifar100":
        train_transforms, test_transforms = get_transforms(args)
        scenario = SplitCIFAR100(
            n_experiences=args.task_nums,
            return_task_id=True,
            dataset_root="./data/cifar100",
            class_ids_from_zero_in_each_exp=True,
            class_ids_from_zero_from_first_exp=False,
            train_transform=train_transforms,
            eval_transform=test_transforms,
        )

        return scenario
    elif args.dataset == "splittinyimagenet":
        train_transforms, test_transforms = get_transforms(args)
        scenario = SplitTinyImageNet(
            n_experiences=args.task_nums,
            return_task_id=True,
            dataset_root="./data/tiny_imagenet",
            class_ids_from_zero_in_each_exp=True,
            class_ids_from_zero_from_first_exp=False,
            train_transform=train_transforms,
            eval_transform=test_transforms,
        )
      
    return scenario


def train(hnet, mnet, scenario, args):
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    hnet.to(device)
    mnet.to(device)
    reg = None
    if args.use_reg:
        reg = Regularizer(args.beta, device)

    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for experience in train_stream:
        task_label = experience.task_label
        train_dataset = experience.dataset
        sub_test_stream = test_stream[: (experience.current_experience + 1)]

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=args.shuffle
        )
        hnet.train()
        mnet.train()


        if task_label > 0 and reg is not None:
            reg_targets = reg.cal_targets(task_id=task_label, hnet=hnet)
        else:
            reg_targets = None

        task_id = torch.Tensor([task_label]).long().to(device)

        for e in tqdm(range(args.epoch)):
            epoch_loss = 0
            epoch_task_loss = 0
            epoch_reg_loss = 0
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                X, Y, _ = batch
                X = X.to(device)
                Y = Y.to(device)

                delta_w = hnet.forward(task_id)
                logits = mnet.forward(X, params=delta_w)
                loss_task = criterion(logits, Y)

                if task_label > 0 and reg_targets is not None:
                    loss_reg = reg.cal_reg(hnet, task_id, reg_targets)
                else:
                    loss_reg = 0

                loss = loss_task + loss_reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_task_loss += loss_task.item()
                epoch_reg_loss += (
                    loss_reg if isinstance(loss_reg, int) else loss_reg.item()
                )

            logging.info(
                f"train experience: {experience.current_experience}, epoch: {e}, loss: {epoch_loss}, task loss: {epoch_task_loss}, reg loss: {epoch_reg_loss}"
            )

        hnet.eval()
        mnet.eval()
        with torch.no_grad():
            acc_list = []
            for test_experience in sub_test_stream:
                test_data_loader = DataLoader(
                    test_experience.dataset, batch_size=args.batchsize
                )
                task_id = torch.Tensor([test_experience.task_label]).long().to(device)
                delta_w = hnet.forward(task_id)

                num_correct = 0
                num_samples = len(test_data_loader.dataset)

                for i, batch in enumerate(test_data_loader):
                    X, Y, _ = batch
                    X = X.to(device)
                    Y = Y.to(device)
                    logits = mnet.forward(X, params=delta_w)
                    num_correct += int(
                        torch.sum(Y == logits.argmax(dim=-1)).detach().cpu()
                    )

                acc = num_correct / num_samples * 100
                acc_list.append(acc)

                logging.info(
                    f"test experience: {test_experience.current_experience}, acc: {acc}"
                )
            logging.info(
                f"test experience: {test_experience.current_experience}, avg acc:{np.mean(acc_list)}"
            )



def run(args):
    log_format = "%(asctime)s - %(message)s"

    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M")

    if args.save_log:
        formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_filename = f"log/{args.dataset}_{args.blocks_fine_tuning}_{args.peft}_{formatted_datetime}_log.txt"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger("").addHandler(file_handler)

    args_as_dict = vars(args)
    params_str = "Parameters:\n" + "\n".join(
        f"{arg_key}: {arg_value}" for arg_key, arg_value in args_as_dict.items()
    )

    logging.info(params_str)

    scenario = load_scenario(args)
    hnet, mnet = load_model(args)
    train(hnet, mnet, scenario, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="splitcifar100",
        choices=["splitmnist", "splitcifar10", "splitcifar100", "splittinyimagenet", "splitimagenet"],
    )
    parser.add_argument("--task_nums", type=int, default=10)
    parser.add_argument("--class_nums", type=int, default=10)

    parser.add_argument(
        "--blocks_fine_tuning",
        type=list,
        default=[0, 1, 2],
        help="blocks for fine tuning",
    )

    parser.add_argument("--peft", type=str, default="lora", choices=["lora", "prefix", "adapter"])
    parser.add_argument("--rank", type=int, default=3, help="rank of LoRAs") 
    parser.add_argument("--prefix_length", type=int, default=5, help="the  length of prefix")  
    parser.add_argument("--down_sample_dims", type=int, default=5, help="the down sample dimension of adapter")  
    parser.add_argument("--use_block_emb", type=bool, default=True, help="whether use block embedding")
    parser.add_argument(
        "--block_emb_sharing",
        type=bool,
        default=False,
        help="whether the block embedding are shared between tasks",
    )
    parser.add_argument("--task_emb_size", type=int, default=8)
    parser.add_argument("--block_emb_size", type=int, default=8)

    parser.add_argument(
        "--use_reg", type=bool, default=True, help="whether to use the reg"
    )
    parser.add_argument("--epoch", type=int, default=10, help="epochs per task")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--shuffle", type=bool, default=True) 
    parser.add_argument("--lr", type=float, default=1e-3, help="lr for Adam")
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--cuda", type=str, default="cuda:7")

    # parser.add_argument("--save_log", action="store_true", help="save log to file")
    parser.add_argument("--save_log", type=bool, default=True, help="save log to file")

    args = parser.parse_args()

    run(args)

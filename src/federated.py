import utils
import models.get_models as get_models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings
from attack.iba import IBA

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="pass in a parameter")
    
    # system
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--rounds", type=int, default=100, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    # 100 for fmnist, 20 for cifar10, 20 for cifar10
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    ) # 0.2 for fmnist, 0.5 for cifar10
    # data
    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--beta", type=float, default=5.0)
    # training
    parser.add_argument(
        "--local_ep", type=int, default=1, help="number of local epochs:E"
    ) # 2 for fmnist, 3 for cifar10
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.05, help="clients learning rate"
    )
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--server_lr", type=float, default=1, help="servers learning rate"
    )
    
    # attack
    parser.add_argument("--reg_conf", type=float, default=0.5, help="the adaptive attack which " \
        "constraint the confidence of output to attack our L_cf. ")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd", "non", "iba"],
    )
    parser.add_argument("--cease_poison", type=float, default=9999)
    parser.add_argument(
        "--num_corrupt", type=int, default=1, help="number of corrupt agents"
    )# 20 for fmnist, 10 for cifar10
    parser.add_argument(
        "--target_class", type=int, default=0, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.1,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )


    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )

    # defense
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "rfa",
            "flame",
            "defender"
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true", help="Whether filtering all malicious clients.")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0) # alignins
    parser.add_argument("--lambda_c", type=float, default=1.0) # alignins
    parser.add_argument("--temperature", type=float, default=2.0, 
                        help='The parameter for knowledge distillation in FedDefender.')
    parser.add_argument(
        "--theta", type=int, default=2, help="break ties when votes sum to 0" # RLR
    )
    
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.1,
        help="the ratio of pruning in lockdown",
    ) # lockdown
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0" # lockdown
    )
    parser.add_argument("--dis_check_gradient", action="store_true", default=False)
    
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    ) # lockdown
    parser.add_argument("--iba_fre", type=int, default=10, 
                        help="the frequency of the training of generator of IBA")

    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.exp_name_extra = "clean"

    if args.super_power:
        args.exp_name_extra = "sp"

    per_data_dict = {
        # "rounds": {"fmnist": 50, "cifar10": 100, "cifar100": 100, "tinyimagenet": 50},
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    # args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)

    # 数据集读取
    train_dataset, val_dataset, normalize = utils.get_datasets(args.data)
    backdoor_train_dataset = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    # 数据集划分
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    # 攻击目标标签 != 原标签，去除本来就是目标标签的样本
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    if args.data != "tinyimagenet":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, normalize=normalize)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args, normalize=normalize
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, normalize=normalize)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
            normalize=normalize
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    global_model = None
    if args.aggr == "defender":
        from defense.feddefender import Agent_defender
        from models.sd_models import get_sd_models
        global_model = get_sd_models(args.data).to(args.device)
    else:
        # initialize a model, and the agents
        global_model = get_models.get_model(args.data, args).to(args.device)

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]
        )
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)

    iba_tool = None
    if args.attack == "iba":
        iba_tool = IBA(args)
    
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                    normalize=normalize
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                    normalize=normalize
                )
        elif args.aggr == "defender" and _id >= args.num_corrupt: # 良性客户端
            agent = Agent_defender(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
                normalize=normalize
            )
        
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
                normalize=normalize
            )
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )

    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector(
            [
                copy.deepcopy(global_model.state_dict()[name])
                for name in global_model.state_dict()
            ]
        )
        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]

        # 使用第一个malicious agent的数据，更新poison data
        if args.attack == "iba" and rnd % args.iba_fre == 0:
            # 只由第一个恶意客户端作为代表更新trigger
            agent_id = 0
            iba_tool.poison_update(
                agent_id, 
                rnd, 
                global_model, 
                agents[agent_id].train_loader, 
                normalization=normalize,
        )
        
        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                    iba_tool=iba_tool
                )
            else:
                #########################################################
                import time
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                start_time = time.time()
                #########################################################
                update = agents[agent_id].local_train(
                    global_model, criterion, rnd, 
                    neurotoxin_mask=neurotoxin_mask,
                    iba_tool=iba_tool
                )
                #########################################################
                torch.cuda.synchronize()
                end_time = time.time()
                mem_after = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()

                print(f"  Time: {end_time - start_time:.4f} s")
                print(f"  Memory Delta: {(mem_after - mem_before) / 1024**2:.2f} MB")
                print(f"  Peak Memory: {peak_mem / 1024**2:.2f} MB")
                #########################################################
                
            # update is the Gredient.
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        updates_dict, neurotoxin_mask = aggregator.aggregate_updates(
            global_model, agent_updates_dict
        )

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:

            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target, normalize=normalize
                )
                    
                if args.attack == "iba":
                    asr = utils.get_loss_n_accuracy(
                        global_model,
                        criterion,
                        val_loader,
                        args,
                        rnd,
                        num_classes=args.num_target,
                        iba_tool=iba_tool, 
                        normalize=normalize
                    )
                    
                    poison_acc = utils.get_loss_n_accuracy(
                        global_model,
                        criterion,
                        val_loader,
                        args,
                        rnd,
                        num_classes=args.num_target,
                        iba_tool=iba_tool,
                        shift_label=False, 
                        normalize=normalize
                    )
                    
                else:    
                    asr = utils.get_loss_n_accuracy(
                        global_model,
                        criterion,
                        poisoned_val_loader,
                        args,
                        rnd,
                        num_classes=args.num_target,
                        normalize=normalize
                    )
                    poison_acc = utils.get_loss_n_accuracy(
                        global_model,
                        criterion,
                        poisoned_val_only_x_loader,
                        args,
                        rnd,
                        args.num_target,
                        normalize=normalize
                    )
                
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target, normalize=normalize
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                    normalize=normalize
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                    normalize=normalize
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc) # acc still remained.

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc
                
                data_split = f'lda{args.beta}' if args.non_iid else 'iid'
                filename = f"{args.data}_{args.attack}_{args.aggr}_resnet101_sp"
                path = './checkpoints/' + filename + '.pth.tar'
                save_dict = {
                    'clean_acc': best_acc,
                    'bad_acc': best_asr,
                    'acr': best_bcdr_acc,
                    'state_dict': global_model.state_dict(),
                }
                torch.save(save_dict, path)
                # 保存iba的攻击模型
                if args.attack == "iba":
                    iba_tool.save_atk_model(name=args.data, path='./checkpoints/')

        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")

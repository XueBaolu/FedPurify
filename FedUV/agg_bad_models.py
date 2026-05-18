from nets.wresnet import wrn_16_2
import torch
import argparse

# 读取两个后门模型
def get_models():
    n_classes = 10
    model_path = "checkpoints/wrn-16-2_m0.5_['squareTrigger', 'gridTrigger']_[0, 0].pth.tar"
    
    net = wrn_16_2(n_classes)
    state_dict = torch.load(model_path)["state_dict"]
    net.load_state_dict(state_dict)
    
    return net


# 测试acc、asr1、asr2
from attack.bad_dataloader import get_bad_dataloader
import copy
from utils.dataloader import partition_data
from utils.dataloader import get_dataloader

def get_test_loaders(args):
    X_train, y_train, X_test, y_test, _, _ = partition_data(
            args,
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.alpha
        )
    _, test_dl, _, _ = get_dataloader(
        args.dataset,
        args.datadir,
        args.batch_size,
        args.batch_size,
        X_train, y_train,
        X_test, y_test
    )
    args_square_0 = copy.deepcopy(args)
    args_square_0.trigger_type = ["squareTrigger"]
    args_square_0.target_label = [0]
    test_loader_square_0 = get_bad_dataloader(
                    args.dataset, 
                    args.batch_size, 
                    None,
                    args_square_0,
                    prop=1.,
                )
    
    args_grid_1 = copy.deepcopy(args)
    args_grid_1.trigger_type = ["gridTrigger"]
    args_grid_1.target_label = [0]
    test_loader_grid_1 = get_bad_dataloader(
                    args.dataset, 
                    args.batch_size, 
                    None,
                    args_grid_1,
                    prop=1.,
                )
    return test_dl, test_loader_square_0, test_loader_grid_1

from utils.calculate_acc import compute_accuracy
def test(global_model, test_loaders, device):
    test_acc, _ = compute_accuracy(global_model, test_loaders[0], get_confusion_matrix=False, device=device)
    square_acc, _ = compute_accuracy(global_model, test_loaders[1], get_confusion_matrix=False, device=device)
    grid_acc, _ = compute_accuracy(global_model, test_loaders[2], get_confusion_matrix=False, device=device)
    print(f"The accs of Agg model: \
          test_acc = {test_acc}, square_acc = {square_acc}, grid_acc = {grid_acc}")

def main(args):
    net = get_models()
    net.to(args.device)
    test_loader, test_loader_square_0, test_loader_grid_1 = get_test_loaders(args)
    test(net, [test_loader, test_loader_square_0, test_loader_grid_1], args.device)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # hyperparameters for system
    parser.add_argument('--seed', type=int, default=42, help='The seed number')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program (cuda/cpu)')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication rounds')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')   
    parser.add_argument('--sample_fraction', type=float, default=0.5, 
                        help='how many clients are sampled in each round')    
    # local training
    parser.add_argument('--epochs', type=int, default=3, help='number of local epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    # model
    parser.add_argument('--model', type=str, default='wrn-16-2', help='neural network used in training')
    parser.add_argument('--simp_width', type=int, default=1, help='multiplier for CNN channel width (only for simple-cnn)')
    parser.add_argument('--out_dim', type=int, default=10, help='the output dimension for the projection layer')
    # save
    parser.add_argument('--logdir', type=str, required=False, default="./logs", help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    # hyperparameters for dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    parser.add_argument('--partition', type=str, required=False, default='iid', help='the data partitioning strategy')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='input batch size for training')
    
    # hyperparameters for algorithm
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='federated learning framework: fedavg/fedprox/moon/freeze/feduv/feddyn')
    parser.add_argument('--mu', type=float, default=0.5, help='the mu parameter for FedProx or MOON')    
    parser.add_argument('--std_coeff', type=float, default=2.5, help='the lambda parameter for FedUV')
    parser.add_argument('--unif_coeff', type=float, default=0.5, help='the mu parameter for FedUV')
    parser.add_argument('--load_first_net', type=int, default=1, 
                        help='whether load the first net as old net or not')
    parser.add_argument('--pool_option', type=str, default='FIFO', 
                        help='whether load the first net as old net or not')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='the temperature parameter for contrastive loss')
    parser.add_argument('--feddyn_alpha', type=float, default=0.1)
    
    # hyperparameters for poisoning
    parser.add_argument('--mode', type=str, default='w', 
                        help='with or without filter, i.e. whether malicious clients would join in the system.')
    parser.add_argument('--m_client_prop', type=float, default=0.5, 
                        help='the proportion of malicious clients in the FL.')
    parser.add_argument('--p_data_prop', type=float, default=0.1,
                        help='the proportion of poisoned samples in malicious\' train dataset.')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', 
                        choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', \
                            'randomPixelTrigger', 'signalTrigger', 'trojanTrigger'])
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--target_type', type=str, default='all2one', 
                        choices=['all2one', 'all2all', 'cleanLabel'])
    parser.add_argument('--trig_w', type=int, default=3)
    parser.add_argument('--trig_h', type=int, default=3)
    parser.add_argument('--lambda_aa', type=float, default=0.1)
    
    
    args = parser.parse_args()
    main(args)
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from easydict import EasyDict
import copy
import random
import numpy as np
import os
from attack.bad_dataloader import get_bad_dataloader
import copy
from utils.dataloader import partition_data
from utils.dataloader import get_dataloader

normalize = None
root = '/home/xuebl/Backdoor_fl/'
ft_lr = 1e-4


def fine_defense_adjust_learning_rate(optimizer, epoch, lr, dataset):
    lr = ft_lr

    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 运行bash生成数据
def run_cmi():
    file_name = "./purify_config.sh"
    # 读取超参数设置
    settings = {}
    with open(file=file_name) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"--([\w\-]+)\s+([^\s\\]+|\"[^\"]*\"|'[^']*')", line)
            if m:
                key, value = m.groups()
                settings[key] = value.strip('"').strip("'")

    return settings


def get_model(settings):
    # 传入后门模型路径
    model_path = root + 'FedUV/var_trigger/' + settings['model_path']

    from nets.wresnet import wrn_16_2
    net = wrn_16_2(settings.num_class)
    checkpoint = torch.load(model_path)
    
    if model_path.endswith('.tar'):
        net.load_state_dict( checkpoint['state_dict'] )
        print("The initial information: ")
        for k, v in checkpoint.items():
            if k != "state_dict":
                print(k, v)
    else:
        net.load_state_dict( checkpoint )
    
    return net.to(settings['device'])


def get_loader(settings):
    # 传入合成数据路径
    test_tf = transforms.Compose([
       transforms.ToTensor(),
    #    normalize,
    ])
    synthetic_data_path = root + 'CMI/' + settings['save_dir']
    syn_dataset = datasets.ImageFolder(root=synthetic_data_path, 
                                       transform=test_tf)
    print("the size of syn_dataset is: ", len(syn_dataset))
    syn_loader = DataLoader(syn_dataset, batch_size=settings.batch_size, shuffle=False)
    
    return syn_loader


def get_train_loader(settings):
    pass


def get_test_loader(args):

    X_train, y_train, X_test, y_test, _, _ = partition_data(
            args, args.dataset, args.datadir, args.log_dir, args.partition, args.n_parties, args.alpha
        )
    
    _, test_dl, _, _ = get_dataloader(
        args.dataset,
        args.datadir,
        args.batch_size,
        args.batch_size,
        X_train, y_train,
        X_test, y_test
    )

    test_loader_dif_triggers = []
    for i in range(args.trigger_type.__len__()):
        args_i = copy.deepcopy(args)
        args_i.trigger_type = [args.trigger_type[i]]
        args_i.target_label = [args.target_label[0]]
        test_loader_i = get_bad_dataloader(
                    args.dataset, 
                    args.batch_size, 
                    None,
                    args_i,
                    prop=1.,
                )
        test_loader_dif_triggers.append(test_loader_i)
    return test_dl, test_loader_dif_triggers

    # args_square_0 = copy.deepcopy(args)
    # args_square_0.trigger_type = [args.trigger_type[0]]
    # args_square_0.target_label = [args.target_label[0]]
    # test_loader_square_0 = get_bad_dataloader(
    #                 args.dataset, 
    #                 args.batch_size, 
    #                 None,
    #                 args_square_0,
    #                 prop=1.,
    #             )
    
    # args_grid_1 = copy.deepcopy(args)
    # args_grid_1.trigger_type = [args.trigger_type[-1]]
    # args_grid_1.target_label = [args.target_label[-1]]
    # test_loader_grid_1 = get_bad_dataloader(
    #                 args.dataset, 
    #                 args.batch_size, 
    #                 None,
    #                 args_grid_1,
    #                 prop=1.,
    #             )
    
    # return test_dl, test_loader_square_0, test_loader_grid_1


# 生成triggers并筛选
def inversion_trigger(settings, target_labels, syn_loader, model):
    if settings.dataset == 'tinyimagenet':
        shape = (3, 64, 64)
    elif settings.dataset == 'fmnist':
        shape = (1, 28, 28)
    else:
        shape = (3, 32, 32) # cifar10, cifar100
    
    from inversion_torch import PixelBackdoor
    inv_triggers = {}
    for label in target_labels:
        print("Processing label: {}".format(label))
        backdoor = PixelBackdoor(model,
                                shape=shape,
                                batch_size=settings.batch_size,
                                normalize=normalize,
                                steps=100,
                                augment=False,
                                device=settings.device)

        pattern = backdoor.generate(syn_loader, label, attack_size=50)
        inv_triggers[label] = pattern
        
    return inv_triggers



# 对比学习净化模型
def purify_model(opt, valid_triggers, model, syn_loader, \
                 test_loader, test_backdoor_dif_triggers):
    
    nets = {'model': model,
            'victimized_model': copy.deepcopy(model),
            'middle_model': None}
    
    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # define loss functions
    criterionCls = nn.CrossEntropyLoss().to(opt.device)

    print('----------- Train Initialization --------------')
    # num_iter = [25, 50]
    num_iter = [10]
    for epoch in range(0, opt.finetuning_epochs):
        # train every epoch
        if epoch in num_iter:
            nets["middle_model"] = copy.deepcopy(model)
        criterions = {'criterionCls': criterionCls}
        print("===Epoch: {}/{}===".format(epoch + 1, opt.finetuning_epochs))
        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)
        train_step(opt, syn_loader, nets, optimizer, criterions, valid_triggers, epoch)
        _, _ = test(opt, backdoor_model, test_loader, test_backdoor_dif_triggers, settings['device'])
        
    return model


def train_step(opt, train_loader, nets, optimizer, criterions, triggers, epoch):

    global normalize

    model = nets['model']
    backup = nets['victimized_model']
    middle = nets['middle_model']

    criterionCls = criterions['criterionCls']
    cos = torch.nn.CosineSimilarity(dim=-1)

    model.train()
    backup.eval()
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.eval()

    flag = -1
    for idx, (data, label) in enumerate(train_loader, start=1):

        data, label = data.clone().to(opt.device), label.clone().to(opt.device)

        negative_data = []
        for _, v in triggers.items():
            neg_data = copy.deepcopy(data)
            neg_data = torch.clamp(neg_data + v, 0, 1)
            negative_data.append(normalize(neg_data))

        data = normalize(data)

        cmi_loss = 0
        for neg_data in negative_data:
            # 后门数据特征
            feature1 = model.get_final_fm(neg_data)
            # 干净数据特征
            feature2 = model.get_final_fm(data)
            # 拉近后门数据和干净数据的特征距离
            posi = cos(feature1, feature2)
            logits = posi.reshape(-1, 1)
            labels = torch.zeros(data.size(0)).to(settings.device).long()
            one_loss = criterionCls(logits, labels)
            cmi_loss = cmi_loss + one_loss
            _, _, output_new, _, _ = model(neg_data)
            relearn_loss = criterionCls(output_new, label)

        loss = - cmi_loss + relearn_loss

        # print(f"sample = {idx+1}, cmi_loss = {cmi_loss.item()}, hold_loss =  {hold_loss.item()}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

from utils.calculate_acc import compute_accuracy
def test(args, model, test_loader, test_backdoor_dif_triggers, device):
    model.eval()
    zip_testloader = [test_loader] + test_backdoor_dif_triggers
    print_name = ['Task Accuracy'] + args.trigger_type
    accs = []

    for i in range(len(zip_testloader)):
        acc, _ = compute_accuracy(model, zip_testloader[i], get_confusion_matrix=False, device=device)
        print(f'{print_name[i]}: {acc * 100}%')
        accs.append(acc)
    
    return accs[0], accs[1]

def addtional_params(settings):
    # 添加其他必要的参数设置
    # system
    # 添加其他必要的参数设置
    # system
    settings["model_path"] = "wrn-16-2_m0.5_['signalTrigger']_[0].pth.tar"
    settings["save_dir"] = "run/cmi_cifar10_signalTrigger"
    settings["trigger_type"] = ['signalTrigger']
    settings["target_label"] = [0]

    if settings['dataset'] == 'cifar10' or settings['dataset'] == 'fmnist':
        settings['num_class'] = 10
    elif settings['dataset'] == 'tinyimagenet':
        settings['num_class'] = 200
    
    settings['ratio'] = 0.02
    settings['batch_size'] = 64
    settings['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # defense
    settings['finetuning_epochs'] = 100
    settings['momentum'] = 0.9
    settings['weight_decay'] = 1e-4
    settings['num_targets'] = 10
    settings['trigger_threshold'] = 0.1
    settings['temperature'] = 0.5

    # system
    settings["datadir"] = root + 'data/'
    settings["log_dir"] = "./logs"
    settings["partition"] = "iid"
    settings["n_parties"] = 20
    settings["alpha"] = 0.9
    settings["target_type"] = "all2one"
    settings["trig_w"] = 3
    settings["trig_h"] = 3

    global normalize
    mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
    std = torch.FloatTensor([0.2023, 0.1994, 0.2010])
    normalize = transforms.Normalize(mean, std)

def set_seed(seed: int = 42):
    # Python 内置随机
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # torch CPU
    torch.manual_seed(seed)

    # torch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保 cudnn 可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 可选：保证 hash 随机性一致
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # prepare
    set_seed(2)
    settings = run_cmi()
    # other params: dataset, poison settings, etc.
    addtional_params(settings)
    print("settings: ", settings)
    settings = EasyDict(settings)
    
    backdoor_model = get_model(settings)
    syn_loader = get_loader(settings)
    # train_loader, train_backdoor_loader = get_train_loader(settings)
    
    test_loader, test_backdoor_dif_triggers = get_test_loader(settings)
    original_clean_acc, original_bad_acc = test(settings, backdoor_model, test_loader, test_backdoor_dif_triggers, settings['device'])
    print(f"== == == Initial ACC, ASR: {original_clean_acc}, {original_bad_acc} == == ==")


    inv_triggers = inversion_trigger(settings, settings['target_label'], syn_loader, backdoor_model)
    # inv_trigger = inv_triggers[settings['target_label']]
    # purify
    purify_model(settings, inv_triggers, backdoor_model, syn_loader, \
                 test_loader, test_backdoor_dif_triggers)
    
    # real data
    # # suspicious_labels = filter_suspicious_targets(settings, test_loader, backdoor_model)
    # inv_triggers = inversion_trigger(settings, [settings['target_label']], test_loader, backdoor_model)
    # inv_trigger = inv_triggers[settings['target_label']]
    # purify_model(settings, inv_trigger, backdoor_model, test_loader)
    # purified_clean_acc, purified_bad_acc = test(backdoor_model, test_loader, test_backdoor_loader, settings['device'])
    
    # save purified model
    model_name = f"purified/{settings['dataset']}/{settings['poison_type']}_t{settings['poison_targets']}.pth"
    torch.save(backdoor_model.state_dict(), root + model_name)
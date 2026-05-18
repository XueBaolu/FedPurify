import subprocess
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, TensorDataset
from easydict import EasyDict
from collections import Counter, defaultdict
from MCL.utils import Normalizer
import copy
import time
import random
import numpy as np
import os

normalize = None
root = './'
ft_lr = 1e-4 # default=1e-4


def fine_defense_adjust_learning_rate(optimizer, epoch, lr, dataset):
    lr = ft_lr

    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 运行bash生成数据
def run_cmi():
    file_name = root + "configs/alignins_imagenet.sh"
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
    model_path = root + settings['model_path']
    # 载入模型
    from MCL.models.selector import select_model
    model = select_model(dataset=settings['dataset'],
                        model_name=settings['model_name'],
                        pretrained=False,
                        pretrained_models_path=None)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if model_path.endswith('.tar'):
        model.load_state_dict( checkpoint['state_dict'] )
        print("The initial information: ")
        for k, v in checkpoint.items():
            if k != "state_dict":
                print(k, v)
    else:
        model.load_state_dict( checkpoint )
    
    return model.to(settings['device'])


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
    syn_loader = DataLoader(syn_dataset, batch_size=settings.batch_size, shuffle=True)

    return syn_loader, None
    
    tf_bd = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Lambda(lambda x: np.array(x))
            ])
    if settings.framework == 'datasetbd':
        # return syn_loader
        syn_dataset_hwc = datasets.ImageFolder(root=synthetic_data_path,
                                                transform=tf_bd)
        from MCL.data_loader import DatasetBD
        syn_backdoor_loader = DataLoader(
            DatasetBD(settings, full_dataset=syn_dataset_hwc, inject_portion=1, \
                    transform=test_tf, mode='test', device=settings.device),
            batch_size=settings.batch_size,
            shuffle=False,
        )
    else:
        import AlignIns.src.utils as utils
        from utils import ImageFolderInMemory
        syn_backdoor_data = ImageFolderInMemory(root=synthetic_data_path, transform=tf_bd)
        # utils.poison_dataset(syn_backdoor_data, settings, poison_all=True, agent_idx=-1, modify_label=False)
        syn_backdoor_loader = DataLoader(syn_backdoor_data, 
                                         batch_size=settings.batch_size, 
                                         shuffle=False)
    
    return syn_loader, syn_backdoor_loader


def get_train_loader(settings):
    if settings['framework'] == 'datasetbd':
        from MCL.data_loader import get_train_loader
        train_loader, train_backdoor_loader = get_train_loader(settings, with_poisoned=True)
    
    elif settings['framework'] == 'alignins':
        from AlignIns.src import utils
        train_dataset, _ = utils.get_datasets(settings["dataset"])
        # from torch.utils.data import Subset
        # rng = np.random.default_rng(10)
        # n_data = len(train_dataset)
        # indices = rng.choice(n_data, size=int(0.05 * n_data), replace=False)
        # train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=5,
            pin_memory=False,)
        train_backdoor_dataset = copy.deepcopy(train_dataset)
        utils.poison_dataset(train_backdoor_dataset, settings, data_idxs=None, poison_all=True, agent_idx=-1, modify_label=False)
        train_backdoor_loader = DataLoader(
            train_backdoor_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=5,
            pin_memory=False,)

    return train_loader, train_backdoor_loader


def get_test_loader(settings):
    # 传入测试数据路径
    test_data_path = root + 'data/'

    # 生成后门数据的两种方式：AlignIns的生成方式，DatasetBD的生成方式
    if settings['framework'] == 'alignins':
        # 测试集读取，投毒
        from AlignIns.src import utils
        import copy
        
        _, val_dataset, data_normalize = utils.get_datasets(settings["dataset"])
        test_loader = DataLoader(
            val_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=5,
            pin_memory=False,)
        
        if settings.attack == 'iba':
            atk_model = None
            from AlignIns.src.attack.atk_model import UNet, MNISTAutoencoder
            if "MNIST" in settings.data.upper(): 
                atk_model = MNISTAutoencoder().to(settings.device)
            else:
                atk_model = UNet(3).to(settings.device)
            
            iba_atk_model_path = "/home/xuebl/Backdoor_fl/AlignIns/src/checkpoints/iba_atk_for_cifar100.pth"
            state_dict = torch.load(iba_atk_model_path, map_location='cpu')
            atk_model.load_state_dict(state_dict)
            atk_model.eval()
            
            all_poisoned_x = []
            all_poisoned_y = []        # 或 logits / feats
            all_y = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(settings.device)
                    poisoned_x = torch.clamp(x + atk_model(x) * 0.3, min=0, max=1)
                    poisoned_y = torch.full_like(y, settings["target_class"][0])

                    all_poisoned_x.append(poisoned_x.cpu())
                    all_poisoned_y.append(poisoned_y)
                    all_y.append(y)
            
            poisoned_X = torch.cat(all_poisoned_x, dim=0)
            poisoned_Y = torch.cat(all_poisoned_y, dim=0)
            original_Y = torch.cat(all_y, dim=0)
            
            poisoned_val_set = TensorDataset(poisoned_X, poisoned_Y)
            test_backdoor_loader = DataLoader(
                poisoned_val_set,
                batch_size=settings.batch_size,
                shuffle=False
            )
            
            poisoned_val_x_set = TensorDataset(poisoned_X, original_Y)
            test_backdoor_x_loader = DataLoader(
                poisoned_val_x_set,
                batch_size=settings.batch_size,
                shuffle=False
            )
            
            return test_loader, test_backdoor_loader, test_backdoor_x_loader
        
        idxs = (val_dataset.targets != settings["target_class"]).nonzero().flatten().tolist()
        if settings["dataset"] != "tinyimagenet":
            poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
            utils.poison_dataset(poisoned_val_set.dataset, settings, idxs, poison_all=True)
            poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
            utils.poison_dataset(
                poisoned_val_set_only_x.dataset,
                settings,
                idxs,
                poison_all=True,
                modify_label=False,
            )
        else:
            poisoned_val_set = utils.DatasetSplit(
                copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=settings
            )
            poisoned_val_set_only_x = utils.DatasetSplit(
                copy.deepcopy(val_dataset),
                idxs,
                runtime_poison=True,
                args=settings,
                modify_label=False,
            )

        test_backdoor_loader = DataLoader(
            poisoned_val_set,
            batch_size=settings['batch_size'],
            shuffle=False,
            num_workers=5,
            pin_memory=False,)

        # 用于测试鲁棒准确度
        test_backdoor_x_loader = DataLoader(
            poisoned_val_set_only_x,
            batch_size=settings['batch_size'],
            shuffle=False,
            num_workers=5,
            pin_memory=False,)

        return test_loader, test_backdoor_loader, test_backdoor_x_loader
        
    elif settings['framework'] == 'datasetbd':
        from MCL.data_loader import get_test_loader
        test_loader, test_backdoor_loader, test_backdoor_x_loader = get_test_loader(settings, with_x_only=True)
    
    else:
        raise NotImplementedError(f"The way of backdoor data generation {settings['framework']} is not supported.")

    return test_loader, test_backdoor_loader, test_backdoor_x_loader



# 生成triggers并筛选
def inversion_trigger(settings, target_labels, syn_loader, model):
    if settings.dataset == 'tinyimagenet':
        shape = (3, 64, 64)
    elif settings.dataset == 'fmnist':
        shape = (1, 28, 28)
    else:
        shape = (3, 32, 32) # cifar10, cifar100
    
    from MCL.inversion_torch import PixelBackdoor
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
        ##################
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        start_time = time.time()
        ##################
        pattern = backdoor.generate(syn_loader, label, attack_size=50)
        ##################
        torch.cuda.synchronize()
        end_time = time.time()
        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Trigger  Time: {end_time - start_time:.4f} s")
        print(f"Trigger  Memory Delta: {(mem_after - mem_before) / 1024**2:.2f} MB")
        print(f"Trigger  Peak Memory: {peak_mem / 1024**2:.2f} MB")
        ##################
        inv_triggers[label] = pattern
        
    return inv_triggers



def purify_with_real_trigger(opt, model, syn_loader, syn_backdoor_loader, \
                             test_loader, test_backdoor_loader, test_backdoor_x_loader):
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
    global normalize
    print('----------- Train Initialization --------------')
    
    for epoch in range(0, opt.finetuning_epochs):
        # train every epoch
        # if epoch == 10:
        #     nets['middle_model'] = copy.deepcopy(model)

        criterions = {'criterionCls': criterionCls}
        print("===Epoch: {}/{}===".format(epoch + 1, opt.finetuning_epochs))
        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)
        

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
        
        for (clean_data, clean_label), (backdoor_data, backdoor_label) in zip(syn_loader, syn_backdoor_loader):
            
            # print("clean labels: ", clean_label)
            # print("backdoor labels: ", backdoor_label)
            clean_data = clean_data.to(opt.device)
            backdoor_data = backdoor_data.to(opt.device)

            clean_data = normalize(clean_data)
            negative_data = normalize(backdoor_data)
            # negative_data = backdoor_data

            feature1 = model.get_final_fm(negative_data)

            feature2 = backup.get_final_fm(clean_data)

            posi = cos(feature1, feature2.detach())
            logits = posi.reshape(-1, 1)

            feature3 = backup.get_final_fm(negative_data)

            nega = cos(feature1, feature3.detach())
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= opt.temperature
            labels = torch.zeros(clean_data.size(0)).to(settings.device).long()
            cmi_loss = criterionCls(logits, labels)


            ## 保留任务知识（基于logits）
            output_new = model(clean_data)
            if middle is not None:
                middle.eval()
                output_ori = middle(clean_data)
            else:
                output_ori = backup(clean_data)

            T = 4.0  # temperature
            with torch.no_grad():
                p_teacher = F.softmax(output_ori / T, dim=1)

            log_p_student = F.log_softmax(output_new / T, dim=1)

            hold_loss = F.kl_div(
                log_p_student,
                p_teacher,
                reduction="batchmean"
            ) * (T * T)


            loss = cmi_loss # + 1.0 * ((0.99) ** epoch) * hold_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, _ = test(backdoor_model, test_loader, test_backdoor_loader, test_backdoor_x_loader, settings['device'])
    
    return model


# 对比学习净化模型
def purify_model(opt, valid_triggers, model, syn_loader, \
                 test_loader, test_backdoor_loader, test_backdoor_x_loader):
    
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
    # num_iter = [20, 50, 100, 200, 300]
    # num_iter = [50, 100, 200, 300]
    num_iter = []
    ##################
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    start_time = time.time()
    ##################
    for epoch in range(0, opt.finetuning_epochs):
        # train every epoch
        if epoch in num_iter:
            nets["middle_model"] = copy.deepcopy(model)
        criterions = {'criterionCls': criterionCls}
        print("===Epoch: {}/{}===".format(epoch + 1, opt.finetuning_epochs))
        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)
        train_step(opt, syn_loader, nets, optimizer, criterions, valid_triggers, epoch)
        _, _ = test(backdoor_model, test_loader, test_backdoor_loader, test_backdoor_x_loader, settings['device'])
        
    ################
    torch.cuda.synchronize()
    end_time = time.time()
    mem_after = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"  Time: {end_time - start_time:.4f} s")
    print(f"  Memory Delta: {(mem_after - mem_before) / 1024**2:.2f} MB")
    print(f"  Peak Memory: {peak_mem / 1024**2:.2f} MB")
    ################
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

        # flag += 1
        # if flag >= 8:
        #     break

        data, label = data.clone().to(opt.device), label.clone().to(opt.device)
        
        negative_data = []
        for _, v in triggers.items():
            neg_data = copy.deepcopy(data)
            neg_data = torch.clamp(neg_data + v, 0, 1)
            negative_data.append(normalize(neg_data))

        data = normalize(data)

        cmi_loss = 0.0
        for neg_data in negative_data:
            # 后门数据特征
            feature1 = model.get_final_fm(neg_data)
            # 干净数据特征
            feature2 = backup.get_final_fm(data)
            # 拉近后门数据和干净数据的特征距离
            posi = cos(feature1, feature2.detach())
            
            ## test，无效
            # if middle is not None:
            #     feature2_m = middle.get_final_fm(data)
            # else:
            #     feature2_m = feature2
            # posi = cos(feature1, feature2_m.detach())

            logits = posi.reshape(-1, 1)

            # 后门数据特征
            feature3 = backup.get_final_fm(neg_data)
            # 拉远后门数据和自身的特征距离
            nega = cos(feature1, feature3.detach())

            ## test，有效
            # if middle is not None:
            #     feature3_m = middle.get_final_fm(neg_data)
            # else:
            #     feature3_m = feature3
            # nega = cos(feature1, feature3_m.detach())
            
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
            logits /= opt.temperature
            labels = torch.zeros(data.size(0)).to(settings.device).long()
            one_trigger_loss = criterionCls(logits, labels)
            cmi_loss = cmi_loss + one_trigger_loss

        
        ## 打印loss
        # posi = cos(feature1, feature2.detach())
        # logits_posi = posi.reshape(-1, 1)
        # logits_posi /= opt.temperature
        # posi_distance = torch.norm(logits_posi, p=2)
        
        # nega = cos(feature1, feature3.detach())
        # logits_nega = nega.reshape(-1, 1)
        # logits_nega /= opt.temperature
        # nega_distance = torch.norm(logits_nega, p=2)
        # print(f"posi_distance = {posi_distance.item()}, nega_distance = {nega_distance.item()}")

        
        ## 保留任务知识（基于logits）
        output_new = model(data)
        if middle is not None:
            middle.eval()
            output_ori = middle(data)
        else:
            output_ori = backup(data)
        # output_ori = backup(data)

        T = 4.0  # temperature
        with torch.no_grad():
            p_teacher = F.softmax(output_ori / T, dim=1)

        log_p_student = F.log_softmax(output_new / T, dim=1)

        hold_loss = F.kl_div(
            log_p_student,
            p_teacher,
            reduction="batchmean"
        ) * (T * T)
        
        ## 保留任务知识（基于features）
        # feature4 = model.get_final_fm(data)
        # hold_loss = F.kl_div(
        #     feature1, 
        #     feature4,
        #     reduction="batchmean"
        # ) * (T * T) ## 过大
        
        # hold_loss = torch.mean(cos(
        #     feature2.detach(), 
        #     feature4,
        #     # reduction="batchmean"
        # ))

        # pred = model(negative_data[0])
        # unlearn_loss = criterionCls(pred, label)
        
        # loss = cmi_loss + 0.1 * ((0.99) ** epoch) * hold_loss
        # loss = cmi_loss + 0.01 * ((0.99) ** epoch) * hold_loss
        # 0.5 for size of synthetic dataset
        loss = cmi_loss + 1.0 * hold_loss
        # print(f"sample = {idx+1}, cmi_loss = {cmi_loss.item()}, hold_loss =  {hold_loss.item()}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader, test_backdoor_loader, test_backdoor_x_loader, device):
    model.eval()
    
    zip_testloader = [test_loader, test_backdoor_loader, test_backdoor_x_loader]
    print_name = ['Task Accuracy', 'Attack Success Rate', 'Robust Accuracy']
    accs = []

    for i in range(3):
        if zip_testloader[i] is None:
            break
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in zip_testloader[i]:
                images, labels = images.to(device), labels.to(device)
                ## aligns 
                images = normalize(images)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'{print_name[i]}: {100 * correct / total}%')
        clean_acc = correct / total
        accs.append(clean_acc)
    
    return accs[0], accs[1]

def addtional_params(settings):
    # 添加其他必要的参数设置
    # system
    # 添加其他必要的参数设置
    # system
    settings['data'] = settings['dataset']
    if settings['dataset'] == 'cifar10' or settings['dataset'] == 'fmnist':
        settings['num_class'] = 10
    elif settings['dataset'] == 'tinyimagenet':
        settings['num_class'] = 200
    
    settings['ratio'] = 0.02
    settings['inject_portion'] = 1
    settings['batch_size'] = 64
    settings['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ## attack: framework 1
    # settings['framework'] = 'datasetbd'
    # settings['attack_method'] = 'badnet'
    # settings['trigger_type'] = 'gridTrigger'
    # settings['target_type'] = 'all2one'
    # settings['target_label'] = 0
    # settings['trig_w'] = 3
    # settings['trig_h'] = 3
    
    ## attack: framework 2
    # settings['framework'] = 'alignins'
    # settings['pattern_type'] = 'plus'
    # settings['attack'] = 'badnet'
    settings['target_class'] = settings['target_label'] = 0
    
    # defense
    settings['finetuning_epochs'] = 100
    settings['momentum'] = 0.9
    settings['weight_decay'] = 1e-4
    settings['num_targets'] = 10
    settings['trigger_threshold'] = 0.1
    settings['temperature'] = 0.5

    global normalize
    normalize = Normalizer(settings['data'])

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
    set_seed(78)
    settings = run_cmi()
    # other params: dataset, poison settings, etc.
    addtional_params(settings)
    print("settings: ", settings)
    settings = EasyDict(settings)
    
    backdoor_model = get_model(settings)
    syn_loader, syn_backdoor_loader = get_loader(settings)
    # train_loader, train_backdoor_loader = get_train_loader(settings)
    
    test_loader, test_backdoor_loader, test_backdoor_x_loader = get_test_loader(settings)
    original_clean_acc, original_bad_acc = test(backdoor_model, test_loader, test_backdoor_loader, test_backdoor_x_loader, settings['device'])
    print(f"== == == Initial ACC, ASR: {original_clean_acc}, {original_bad_acc} == == ==")


    inv_triggers = inversion_trigger(settings, [settings['target_label']], syn_loader, backdoor_model)
    # inv_trigger = inv_triggers[settings['target_label']]
    # purify
    purify_model(settings, inv_triggers, backdoor_model, syn_loader, \
                 test_loader, test_backdoor_loader, test_backdoor_x_loader)

    # purify_with_real_trigger(settings, backdoor_model, test_loader, test_backdoor_loader, \
    #                          test_loader, test_backdoor_loader, test_backdoor_x_loader)   
    
    # purify_with_real_trigger(settings, backdoor_model, syn_loader, syn_backdoor_loader, \
    #                          test_loader, test_backdoor_loader, test_backdoor_x_loader)
    
    # real data
    # # suspicious_labels = filter_suspicious_targets(settings, test_loader, backdoor_model)
    # inv_triggers = inversion_trigger(settings, [settings['target_label']], test_loader, backdoor_model)
    # inv_trigger = inv_triggers[settings['target_label']]
    # purify_model(settings, inv_trigger, backdoor_model, test_loader)
    # purified_clean_acc, purified_bad_acc = test(backdoor_model, test_loader, test_backdoor_loader, settings['device'])
    
    # save purified model
    model_name = f"purified/{settings['dataset']}/{settings['poison_type']}_t{settings['poison_targets']}.pth"
    torch.save(backdoor_model.state_dict(), root + model_name)
    
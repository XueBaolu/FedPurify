import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import OrderedDict
import torch.nn.functional as F
import copy
from torch.nn.utils import parameters_to_vector
import time
from agent import Agent
from utils import vector_to_model_wo_load


class Agent_defender(Agent):
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None, normalize=None):
        super().__init__(id, args, train_dataset, data_idxs, mask, backdoor_train_dataset, normalize)
    
    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None, iba_tool=None):
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        
        global_model.train()
        initial_global_model = copy.deepcopy(global_model)
        initial_global_model.eval()
        
        optimizer = torch.optim.SGD(global_model.parameters(), 
                                    lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)
        
        cos=torch.nn.CosineSimilarity(dim=-1).to(self.args.device)
        kl_criterion = nn.KLDivLoss(reduction="batchmean").to(self.args.device)
        
        
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                labels.to(device=self.args.device, non_blocking=True)
                
                if self.normalize is not None:
                        inputs = self.normalize(inputs)

                outputs, SD_outputs,  feats = global_model(inputs, get_feat=True, SD=True)
                SD_p_output = F.softmax(SD_outputs / self.args.temperature, dim=1)
                SD_logp = F.log_softmax(SD_outputs / self.args.temperature, dim=1)
                p_output = F.softmax(outputs / self.args.temperature,dim=1)
                logp_output = F.log_softmax(outputs / self.args.temperature,dim=1)
                
                with torch.no_grad():
                    logp_global = initial_global_model(inputs) 
                    logp_global = F.softmax(logp_global / self.args.temperature, dim=1)
                    logp_global = logp_global.detach()
                
                alpha = cos(logp_global, 
                            F.one_hot(labels, num_classes=self.args.num_target)
                            ).unsqueeze(1)
                targer_g = (1-alpha) * F.one_hot(labels, num_classes=self.args.num_target) \
                    + alpha * logp_global
                loss_gkd = -torch.mean(torch.sum(SD_logp* targer_g, dim=1))
                loss = criterion(outputs, labels) + loss_gkd + kl_criterion(logp_output, SD_p_output.detach())
                loss.backward(retain_graph=True)
                
                targets_fast = labels.clone()
                randidx = torch.randperm(labels.size(0))
                for n in range(int(labels.size(0)*0.5)):
                    num_neighbor = 10
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1,feat.size(0))
                    feat.data = feat.data.expand(labels.size(0),feat.size(0))
                    dist = torch.sum((feat-feats)**2,dim=1)
                    _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                    targets_fast[idx] = labels[neighbor[random.randint(1,num_neighbor)]]

                fast_loss = criterion(outputs,targets_fast)
                grads = torch.autograd.grad(fast_loss, global_model.parameters(), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

                for grad in grads:
                    if grad == None:
                        continue
                    grad = grad.detach()
                    grad.requires_grad = False  

                fast_weights = OrderedDict(
                    (name, param - self.args.client_lr*grad) 
                    for ((name, param), grad) in zip(global_model.named_parameters(), grads) 
                    if grad !=None)
                fast_out, SD_fast_out = global_model(inputs, fast_weights, SD=True)  

                logp_fast = F.log_softmax(fast_out, dim=1)
                meta_loss = criterion(fast_out, labels)
                meta_loss.backward()
                optimizer.step()


                # if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                #     if self.args.data == 'cifar10':
                #         eps = torch.norm(initial_global_model_params) * 0.1
                #     else:
                #         eps = torch.norm(initial_global_model_params)

                #     current_local_model_params = parameters_to_vector([net.state_dict()[name] for name in net.state_dict()]).detach()
                #     norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
                #     print('clip before: ', norm_diff)
                #     if norm_diff > eps:
                #         w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                #         print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                #         new_state_dict = vector_to_model_wo_load(w_proj_vec, initial_global_model)    
                #         global_model.load_state_dict(new_state_dict)

            
        end = time.time()
        train_time = end - start
        print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t meta_loss: %.8f \t time: %.2f" % \
            (local_epoch, self.id, str(self.is_malicious), loss, meta_loss, train_time))


        with torch.no_grad():
            # global_model.load_state_dict(net.state_dict())
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params

            return self.update

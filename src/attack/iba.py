import torch.nn as nn
import torch
import os
import time
import shutil
from .atk_model import UNet, MNISTAutoencoder


DEFAULT_PARAMS = {
    "atk_eps": 0.3,
    "atk_test_eps": 0.05,  # Target epsilon after decay
    "eps_decay_rate": 0.01,  # Decay rate per round
    "atk_lr": 0.01,
    "outter_epochs": 200,
    "save_atk_model_at_last": True,
}

class IBA():
    def __init__(self, args):
        
        self.args = args
        self.sync_poison = True  # Sync poison resources across clients
        self.save_atk_model_at_last = False

        # Initialize local model
        if "MNIST" in self.args.data.upper(): 
            self.atk_model = MNISTAutoencoder().to("cuda")
            self.atk_model_name = "mnist_autoencoder"
        else:
            self.atk_model = UNet(3).to("cuda")    
            self.atk_model_name = "unet"

        if self.save_atk_model_at_last:
            self.atk_model_path = os.path.join("backfed/poisons/saved", "iba")
            os.makedirs(self.atk_model_path, exist_ok=True)
        
        # Epsilon decay tracking
        self.cur_eps = DEFAULT_PARAMS["atk_eps"]  # Current epsilon
        self.decay_start_round = None  # Track when decay starts
    
    def exponential_decay(self, init_val, decay_rate, t):
        """Exponential decay: init_val * (1 - decay_rate)^t"""
        return init_val * (1.0 - decay_rate) ** t
    
    def update_epsilon(self, server_round):
        """Update current epsilon with decay"""
        if self.decay_start_round is None:
            self.decay_start_round = server_round
        
        t = server_round - self.decay_start_round
        decayed_eps = self.exponential_decay(DEFAULT_PARAMS["atk_eps"], DEFAULT_PARAMS["eps_decay_rate"], t)
        self.cur_eps = max(DEFAULT_PARAMS["atk_test_eps"], decayed_eps)
        

    @torch.no_grad()
    def poison_inputs(self, inputs):
        self.atk_model.eval()
        noise = self.atk_model(inputs) * self.cur_eps
        return torch.clamp(inputs + noise, min=0, max=1)
    
    def poison_update(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """Update the trigger generator model"""
        # Update epsilon with decay
        self.update_epsilon(server_round)
        self.train_atk_model(client_id=client_id, 
                            server_round=server_round, 
                            model=initial_model, 
                            dataloader=dataloader, 
                            normalization=normalization)

    def train_atk_model(self, client_id, server_round, model, dataloader, normalization=None):
        start_time = time.time()

        # if len(dataloader) == 0:
        #     return self.trigger_image.detach()

        loss_fn = nn.CrossEntropyLoss()
        # training trigger
        model.eval()  # classifier model
        self.freeze_model(model)
        
        self.atk_model.train()  # trigger model
        num_attack_sample = -1  # poison all samples

        local_asr, threshold_asr = 0.0, 0.85  # Stop training if local ASR exceeds threshold
        atk_optimizer = torch.optim.Adam(self.atk_model.parameters(), lr=DEFAULT_PARAMS["atk_lr"])
        
        for atk_train_epoch in range(DEFAULT_PARAMS["outter_epochs"]):
            if local_asr >= threshold_asr:
                print(f"Client [{client_id}]: Early stopping - threshold_asr reached \
                    ({local_asr:.4f} >= {threshold_asr})")
                break

            backdoor_preds, backdoor_loss, total_sample = 0, 0, 0
            
            for _, batch in enumerate(dataloader):
                inputs, labels = batch[0].to("cuda"), batch[1].to("cuda")
                
                # Zero gradients for the optimizer
                atk_optimizer.zero_grad()
                
                # inputs = normalization(inputs)
                # Generate poisoned inputs using the attack model
                noise = self.atk_model(inputs) * self.cur_eps
                poisoned_inputs = torch.clamp(inputs + noise, min=0, max=1)
                poisoned_labels = self.poison_labels(labels)
                
                if normalization:
                    poisoned_inputs = normalization(poisoned_inputs)

                if num_attack_sample != -1:
                    poisoned_inputs = poisoned_inputs[:num_attack_sample]
                    poisoned_labels = poisoned_labels[:num_attack_sample]
                
                # Forward pass through the classifier model
                poisoned_outputs = model(poisoned_inputs)
                loss_p = loss_fn(poisoned_outputs, poisoned_labels)
                backdoor_loss += loss_p.item()
                
                # Backward pass
                loss_p.backward()
                atk_optimizer.step()

                backdoor_preds += (torch.max(poisoned_outputs.data, 1)[1] == poisoned_labels).sum().item()
                total_sample += len(poisoned_labels)

            local_asr = backdoor_preds / total_sample
            backdoor_loss = backdoor_loss / len(dataloader)
            if atk_train_epoch % 10 == 0:
                print(f"Epoch {atk_train_epoch} updated atk_model - local_asr: {local_asr*100:.2f}% | threshold_asr: {threshold_asr*100:.2f}% | backdoor_loss: {backdoor_loss}")
        
        self.unfreeze_model(model)
        end_time = time.time()


    def save_atk_model(self, name, path=None):
        """
        Save the attacker model for the poisoning round and keep track of the latest version.
        """

        save_path = os.path.join(path, f"iba_atk_for_{name}.pth")
        torch.save(self.atk_model.state_dict(), save_path)
    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
    def get_shared_resources(self) -> dict:
        """
        Get the resources to be shared across clients in parallel mode.
        Returns:
            resources (dict): The resources to be shared
        """
        return {
            "atk_model_state_dict": {k: v.cpu() for k, v in self.atk_model.state_dict().items()},
            "decay_start_round": self.decay_start_round
        }
    
    def update_shared_resources(self, resources: dict):
        """
        Update the resources shared across clients in parallel mode.
        Args:
            resources (dict): The resources to be updated
        """
        self.atk_model.load_state_dict(resources["atk_model_state_dict"])
        self.decay_start_round = resources["decay_start_round"]

    def poison_finish(self):
        if self.save_atk_model_at_last:
            self.save_atk_model(name=self.atk_model_name)
            
    
    def poison_labels(self, labels):
        """
        Return the poisoned labels.
        Args:
            labels (torch.Tensor or int): Labels to poison. Can be a tensor or a single integer.
            source_target_mappings (dict): Source-target mappings for the labels
            test (bool): Whether to poison the entire batch or a portion of it
        Return:
            poisoned_labels (torch.Tensor or int): Poisoned labels in the same format as input
        """
        # Handle scalar input (int or 0-dim tensor)
        is_scalar = isinstance(labels, int) or (torch.is_tensor(labels) and labels.dim() == 0)

        if is_scalar:
            # Handle scalar inputs directly
            return self.args.target_class

        # Handle tensor inputs
        return torch.ones(len(labels), dtype=torch.long, device=self.args.device) * self.args.target_class
from copy import deepcopy
import torch
import numpy as np
import hdbscan
from utils import vector_to_model_wo_load
from torch.nn.utils import parameters_to_vector

def normclipping(vectors, threshold, epsilon=1e-6):
    """ clipping the 2d-vectors based on the threshold
    Args:
        2d vectors (numpy.ndarray): the vectors from clients
    """
    if len(vectors.shape) != 2:
        raise ValueError(
            "The input should be 2d vectors, or you need to extend this function")
    return vectors * np.minimum(1, threshold / (np.linalg.norm(vectors, axis=1)+epsilon)).reshape(-1, 1)


def addnoise(vector, noise_mean, noise_std):
    """ add gaussian noise to the vector, z~N(0, sigma^2 * I)
    """
    # generate gaussian noise, note that the noise should be float32 to be consistent with the future torch dtype
    noise = np.random.normal(noise_mean, noise_std,
                            vector.shape).astype(np.float32)
    return vector + noise


def prepare_updates(updates, global_model):
    num_updates = len(updates)  # equal to num_clients
    gradient_updates = updates
    global_vector = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
    vec_updates = torch.stack([
        global_vector + updates[cid] 
        for cid in range(num_updates)
    ])

    # vector_form return 1d np array vector model parameters
    model_updates = vec_updates
    
    return model_updates, gradient_updates

class FLAME():
    """
    [FLAME: Taming Backdoors in Federated Learning](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) - USENIX Security '22
    FLAME first clusters the cosine distance between client updates with hdbscan, then clips the benign gradients by the median of norms, and finally adds noise to meet the requirements of differential privacy.
    """

    def __init__(self, global_model, num_clients):
        self.gamma = 1.2e-5 # 1.2e-5 for fmnist
        self.global_model = global_model
        self.num_clients = num_clients

    def aggregate(self, updates):
        model_updates, gradient_updates = prepare_updates(updates, self.global_model)
        np_model_updates = model_updates.detach().cpu().numpy()
        np_gradient_updates = np.stack([
            u.detach().cpu().numpy()
            for u in gradient_updates
        ])
        benign_idx = self.cosine_clustering(np_model_updates)
        aggregated_gradient, median_norm = self.adpative_clipping(np_gradient_updates, benign_idx)
        gradient = torch.from_numpy(aggregated_gradient)\
            .to(device=gradient_updates[0].device, dtype=gradient_updates[0].dtype)
        
        global_updates = self.add_noise2gredient(self.gamma * median_norm, gradient)

        
        return global_updates

    def cosine_clustering(self, model_updates):
        """
        clustering the cosine distance between client updates with hdbscan
        """
        cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                                min_cluster_size=self.num_clients//2+1, min_samples=1, allow_single_cluster=True)
        cluster.fit(model_updates.astype(np.float64))
        # choose which cluster is benign
        return [idx for idx, label in enumerate(cluster.labels_) if label == 0]

    def adpative_clipping(self, gradient_updates, benign_idx):
        """
        clipping threshold is the median of l2 distance between last global model and current clients updates
        """
        # 1. get median of l2 norm
        median_norm = np.median(np.linalg.norm(gradient_updates, axis=1))
        # 2. clip the benign gradients by median of norms
        clipped_gradient_updates = normclipping(
            gradient_updates[benign_idx], median_norm)
        # 3. calculate the mean of clipped benign gradient updates and add them to the last global model for aggregation
        aggregated_gradient = np.mean(clipped_gradient_updates, axis=0)
        
        return aggregated_gradient, median_norm
    
        
    def add_noise2gredient(self, noise_scale, gredient, only_weights=True):
        # add gaussian noise to the aggregated grdient
        model_state_dict = vector_to_model_wo_load(gredient, self.global_model)
        for key, param in model_state_dict.items():
            if only_weights:
                if any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
                    continue
            std = noise_scale * param.data.std()
            noise = torch.normal(
                mean=0, std=std, size=param.size()).to(param.device)
            param.data += noise
        
        noised_gredient = parameters_to_vector([model_state_dict[name] for name in model_state_dict.keys()])
        return noised_gredient
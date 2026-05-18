A benchmark of data-free backdoor purification in Federated Learning from paper ["FedPurify: Knowledge-Preserving Backdoor Defense with
Data-Free Purification in Federated Learning"]

## Quick Start

### 1. FL training
Run federated learning(FL) training to obtain a converged global model under different backdoor attack settings. 
You can perform FL training using the AlignIns framework: 

```bash
cd AlignIns/src
bash run_fmnist.sh
```
Alternatively, you can use the FedUV framework to support non-IID FL algorithms:

```bash
cd FedUV
python train_fl.py --dataset cifar10 --model wrn-16-2 --alg fedavg --trigger_type trojanTrigger
```

### 2. Data Generation

Then, synthetic data are generated based on the converged backdoored model to prepare for the subsequent purification process.

```bash
cd CMI
bash scripts/cmi/cmi_cifar10_for_vis.sh
```

**Note:** Please configure the hyperparameters in the corresponding `.sh` file before running the script.

| Argument | Description |
|---|---|
| `dataset` | Dataset used for the main task. |
| `teacher` | Model architecture used in FL. |
| `teacher_path` | Path to the saved parameters of the backdoored model. |
| `save_dir` | Directory for saving the generated synthetic data. |

### 3. Model Purification

Finally, run `fed_purify.py` to purify the converged backdoored global model.
Before execution, please configure the required hyperparameters in the `configs/` directory.

```bash

python fed_purify.py
```

## Citation
If you found this work useful for your research, please cite our paper:
```
@article{,
  title={FedPurify: Knowledge-Preserving Backdoor Defense with Data-Free Purification in Federated Learning},
  author={Baolu Xue, Hanyuan Zheng, Tianxing Man, Bing Chen},
  journal={},
  year={2026}
}
```

## Reference
* AlignIns: [Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](https://arxiv.org/abs/2503.07978)
* FedUV: [Feduv: uniformity and variance for heterogeneous federated learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Son_FedUV_Uniformity_and_Variance_for_Heterogeneous_Federated_Learning_CVPR_2024_paper.pdf)
* CMI: [Contrastive model inversion for data-free knowledge distillation](https://arxiv.org/pdf/2105.08584)
* MCL: [Model-Contrastive Learning for Backdoor Elimination](https://dl.acm.org/doi/pdf/10.1145/3581783.3612415)

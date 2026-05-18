#!/home/user/Workspace/Backdoor_fl/CMI/ bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$PROJECT_ROOT"

python datafree_kd.py \
--method cmi \
--dataset tiny_imagenet \
--batch_size 64 \
--teacher resnet101_imagenet \
--teacher_path "Backdoor_fl/AlignIns/src/checkpoints/tinyimagenet_badnet_avg_resnet101.pth.tar" \
--save_dir run/cmi_resnet101_1 \
--log_tag fedavg \
--student resnet18 \
--lr 0.1 \
--kd_steps 1 \
--ep_steps 1 \
--epochs 1 \
--g_steps 400 \
--lr_g 2e-4 \
--adv 0. \
--bn 1.0 \
--oh 0.5 \
--cr 0.8 \
--H 0.0 \
--cr_T 0.1 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 6 \
--T 20 \
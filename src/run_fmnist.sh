
# attack：choices=["badnet", "DBA", "neurotoxin", "iba"]
# aggr：  choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", "foolsgold", "rfa", "flame", "defender"]

python federated.py \
--device cuda:0 \
--data fmnist \
--rounds 100 \
--client_lr 0.05 \
--num_agents 100 \
--agent_frac 0.2 \
--local_ep 2 \
--attack badnet \
--num_corrupt 20 \
--aggr avg \
--super_power


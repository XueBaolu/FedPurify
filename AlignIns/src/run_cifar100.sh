# attack choices: ["badnet", "DBA", "neurotoxin", "iba"]
# defense choices: ["avg", "alignins", "rlr", "lockdown", "flame", "defender"]

python federated.py \
--data cifar100 \
--rounds 100 \
--num_agents 20 \
--agent_frac 0.5 \
--local_ep 3 \
--attack iba \
--num_corrupt 10 \
--aggr avg \
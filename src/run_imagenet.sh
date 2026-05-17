# attack：choices=["badnet", "DBA", "neurotoxin", "iba"]
# aggr：  choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", "foolsgold", "rfa", "flame", "defender"]
attack=$1

python federated.py \
--data tinyimagenet \
--rounds 100 \
--num_agents 20 \
--agent_frac 0.5 \
--local_ep 2 \
--attack $attack \
--num_corrupt 5 \
--aggr alignins \
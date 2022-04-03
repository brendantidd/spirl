'''
Preparing a list of commands for eventually running all through a script.
Notes:
- Kitchen task was state based
'''

# Probably easier just to collect the data using the skilled_residual code and point to the directory.
# Collect data, from skilled_residuals/src/utils:
python3 collect_demos_fetch.py
# Data directory:
~/skilled_residuals/data

# Training skilled priors and enc/dec - hierarchical closed loop
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/fetch_stacking/hierarchical_cl --val_data_size=160 --gpu 0


# Training RL agent:
python3 spirl/rl/train.py --path=spirl/configs/hrl/fetch_stacking/spirl_cl --seed=0 --prefix=SPIRL_fetch_stacking_seed0 

# Baseline commands:

# Single-step action prior:
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/fetch_stacking/flat --val_data_size=160

# Vanilla SAC
python3 spirl/rl/train.py --path=spirl/configs/rl/fetch_stacking/prior_initialized/flat_prior/ --seed=0 --prefix=flatPrior_fetch_stacking_seed0

# SAC w/ single-step action prior
python3 spirl/rl/train.py --path=spirl/configs/rl/fetch_stacking/prior_initialized/flat_prior/ --seed=0 --prefix=flatPrior_fetch_stacking_seed0

# BC + finetune
python3 spirl/rl/train.py --path=spirl/configs/rl/fetch_stacking/prior_initialized/bc_finetune/ --seed=0 --prefix=bcFinetune_fetch_stacking_seed0

# Skill Space Policy w/o prior
python3 spirl/rl/train.py --path=spirl/configs/hrl/fetch_stacking/no_prior/ --seed=0 --prefix=SSP_noPrior_fetch_stacking_seed0


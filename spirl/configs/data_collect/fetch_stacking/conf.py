import os

from spirl.utils.general_utils import AttrDict

# from spirl.data.fetch_stacking.src.demo_gen.fetch_stacking_demo_agent import FetchStackingDemoAgent
# from spirl.data.fetch_stacking.src.fetch_stacking_env import FetchStackEnv
# from spirl.data.fetch_stacking.src.fetch_task_generator import FixedSizeSingleTowerFetchTaskGenerator

# from skill_residuals.src.utils.collect_skills_prior_dataset import CollectSkillsDataset
from skill_residuals.src.utils.collect_demos_fetch import CollectDemos

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used for generating fetch stacking dataset'
SEED = 31

configuration = {
    'seed': SEED,
    'agent': CollectDemos,
    'environment': FetchStackEnv,
    'max_rollout_len': 250,
}
configuration = AttrDict(configuration)

# Task
task_params = AttrDict(
    max_tower_height=4,
    seed=SEED,
)

# Agent
agent_config = AttrDict(

)

# Dataset - Random data
data_config = AttrDict(

)

# Environment
env_config = AttrDict(
    task_generator=FixedSizeSingleTowerFetchTaskGenerator,
    task_params=task_params,
    dimension=2,
    n_steps=2,
    screen_width=32,
    screen_height=32,
    rand_task=True,
    rand_init_pos=True,
    camera_name='agentview',
)


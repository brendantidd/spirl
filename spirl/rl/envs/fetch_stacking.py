import numpy as np

from spirl.rl.components.environment import GymEnv
# from spirl.rl.envs.fetch_stacking.fetch_stack_env_rl import FetchStackEnv
from skilled_residuals.rl.envs.fetch_stacking.fetch_stack_env_rl import FetchStackEnv
from spirl.utils.general_utils import AttrDict, ParamDict

class FetchStackEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        self.fetch_stack = FetchStackEnv(*args, **kwargs)
        super.__init__()

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "fetch-mixed-v0",
        }))

    def reset(self):
        return self.fetch_stack.reset()

    def step(self):
        return self.fetch_stack.step()


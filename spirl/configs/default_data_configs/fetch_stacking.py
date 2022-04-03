from spirl.utils.general_utils import AttrDict
# from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.components.data_loader import GlobalSplitDataset

data_spec = AttrDict(
    dataset_class=GlobalSplitDataset,
    n_actions=4,
    state_dim=19,
    split=AttrDict(train=0.95, val=0.05, test=0.0),
    res=32,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 150

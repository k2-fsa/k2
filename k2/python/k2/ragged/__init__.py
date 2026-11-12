# please sort imported functions alphabetically
import torch
from _k2.ragged import RaggedShape
from _k2.ragged import RaggedTensor
from _k2.ragged import cat
from _k2.ragged import create_ragged_shape2
from _k2.ragged import create_ragged_tensor
from _k2.ragged import index
from _k2.ragged import index_and_sum
from _k2.ragged import random_ragged_shape
from _k2.ragged import regular_ragged_shape

# Since PyTorch 2.6, torch.load() defaults to weights_only=True, which refuses
# to unpickle classes that are not explicitly allowlisted. The pickled state of
# RaggedTensor contains only tensors and strings, so allowlisting it is safe
# and lets torch.load() restore RaggedTensor instances saved by torch.save().
# add_safe_globals() is available since PyTorch 2.4.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([RaggedTensor])

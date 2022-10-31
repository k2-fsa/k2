from pathlib import Path as _Path

import torch  # noqa
from .torch_version import k2_torch_cuda_version
from .torch_version import k2_torch_version

if torch.__version__.split("+")[0] != k2_torch_version.split("+")[0]:
    raise ImportError(
        f"k2 was built using PyTorch {k2_torch_version}\n"
        f"But you are using PyTorch {torch.__version__} to run it"
    )

if (
    k2_torch_cuda_version != ""
    and torch.version.cuda is not None
    and torch.version.cuda != k2_torch_cuda_version
):
    raise ImportError(
        f"k2 was built using CUDA {k2_torch_cuda_version}\n"
        f"But you are using CUDA {torch.version.cuda} to run it."
    )

try:
    from _k2 import DeterminizeWeightPushingType
    from _k2 import simple_ragged_index_select
except ImportError as e:
    import sys

    major_v, minor_v = sys.version_info[:2]
    raise ImportError(
        str(e) + "\nNote: If you're using anaconda and importing k2 on MacOS,"
        "\n      you can probably fix this by setting the environment variable:"
        f"\n  export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib/python{major_v}.{minor_v}/site-packages:$DYLD_LIBRARY_PATH"  # noqa
    )
from .ragged import RaggedShape
from .ragged import RaggedTensor

from . import autograd
from . import autograd_utils
from . import dense_fsa_vec
from . import fsa
from . import utils

#
from .autograd import intersect_dense
from .autograd import intersect_dense_pruned
from .ctc_loss import CtcLoss
from .ctc_loss import ctc_loss
from .dense_fsa_vec import DenseFsaVec
from .dense_fsa_vec import convert_dense_to_fsa_vec
from .fsa import Fsa
from .fsa_algo import add_epsilon_self_loops
from .fsa_algo import arc_sort
from .fsa_algo import closure
from .fsa_algo import compose
from .fsa_algo import connect
from .fsa_algo import ctc_graph
from .fsa_algo import ctc_topo
from .fsa_algo import determinize
from .fsa_algo import expand_ragged_attributes
from .fsa_algo import intersect
from .fsa_algo import intersect_device
from .fsa_algo import invert
from .fsa_algo import levenshtein_alignment
from .fsa_algo import levenshtein_graph
from .fsa_algo import linear_fsa
from .fsa_algo import linear_fsa_with_self_loops
from .fsa_algo import linear_fst
from .fsa_algo import linear_fst_with_self_loops
from .fsa_algo import prune_on_arc_post
from .fsa_algo import random_paths
from .fsa_algo import remove_epsilon
from .fsa_algo import remove_epsilon_and_add_self_loops
from .fsa_algo import remove_epsilon_self_loops
from .fsa_algo import replace_fsa
from .fsa_algo import reverse
from .fsa_algo import shortest_path
from .fsa_algo import top_sort
from .fsa_algo import trivial_graph
from .fsa_algo import union
from .fsa_properties import to_str as properties_to_str
from .mutual_information import joint_mutual_information_recursion
from .mutual_information import mutual_information_recursion
from .nbest import Nbest
from .ops import cat
from .ops import compose_arc_maps
from .ops import index_add
from .ops import index_fsa
from .ops import index_select

from .rnnt_decode import RnntDecodingConfig
from .rnnt_decode import RnntDecodingStream
from .rnnt_decode import RnntDecodingStreams

from .rnnt_loss import do_rnnt_pruning
from .rnnt_loss import get_rnnt_logprobs
from .rnnt_loss import get_rnnt_logprobs_joint
from .rnnt_loss import get_rnnt_logprobs_pruned
from .rnnt_loss import get_rnnt_logprobs_smoothed
from .rnnt_loss import get_rnnt_prune_ranges
from .rnnt_loss import get_rnnt_prune_ranges_deprecated  # for testing purpose
from .rnnt_loss import rnnt_loss
from .rnnt_loss import rnnt_loss_pruned
from .rnnt_loss import rnnt_loss_simple
from .rnnt_loss import rnnt_loss_smoothed

from .symbol_table import SymbolTable
from .utils import create_fsa_vec
from .utils import create_sparse
from .utils import is_rand_equivalent
from .utils import get_best_matching_stats
from .utils import to_dot
from .utils import to_str
from .utils import to_str_simple
from .utils import to_tensor
from .utils import random_fsa
from .utils import random_fsa_vec
from _k2.version import with_cuda

cmake_prefix_path = _Path(__file__).parent / "share" / "cmake"
del _Path

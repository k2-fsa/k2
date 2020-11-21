from .autograd import get_tot_scores
from .autograd import index_select
from .autograd import intersect_dense_pruned
from .autograd import union
from .dense_fsa_vec import DenseFsaVec
from .dense_fsa_vec import convert_dense_to_fsa_vec
from .fsa import Fsa
from .fsa_algo import add_epsilon_self_loops
from .fsa_algo import arc_sort
from .fsa_algo import connect
from .fsa_algo import determinize
from .fsa_algo import intersect
from .fsa_algo import linear_fsa
from .fsa_algo import remove_epsilon
from .fsa_algo import shortest_path
from .fsa_algo import top_sort
from .fsa_properties import to_str as properties_to_str
from .ops import index
from .ops import index_add
from .symbol_table import SymbolTable
from .utils import create_fsa_vec
from .utils import is_rand_equivalent
from .utils import to_dot
from .utils import to_str
from .utils import to_tensor


def use_cuda() -> bool:
    import _k2
    return _k2.version.use_cuda


# please keep the list sorted
__all__ = [
    'DenseFsaVec',
    'Fsa',
    'SymbolTable',
    'add_epsilon_self_loops',
    'arc_sort',
    'connect',
    'convert_dense_to_fsa_vec',
    'create_fsa_vec',
    'determinize',
    'get_tot_scores',
    'index',
    'index_add',
    'index_select',
    'intersect',
    'intersect_dense_pruned',
    'linear_fsa',
    'properties_to_str'
    'properties_to_str',
    'remove_epsilon',
    'shortest_path',
    'to_dot',
    'to_str',
    'to_tensor',
    'top_sort',
    'union',
    'use_cuda',
]

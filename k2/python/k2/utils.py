# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from .fsa import Fsa
from _k2 import _fsa_to_str


def to_str(fsa: Fsa, openfst: bool = False) -> str:
    '''Convert an Fsa to a string.

    Note:
      The returned string can be used to construct an Fsa.

    Args:
      openfst:
        Optional. If true, we negate the score during the conversion,

    Returns:
      A string representation of the Fsa.
    '''
    if hasattr(fsa, 'aux_labels'):
        aux_labels = fsa.aux_labels
    else:
        aux_labels = None
    return _fsa_to_str(fsa.arcs, openfst, aux_labels)


def to_dot(fsa) -> 'Digraph':
    '''Visualize an Fsa via graphviz.
    '''
    from graphviz import Digraph
    assert len(fsa.shape) == 2, 'FsaVec is not supported'
    if hasattr(fsa, 'aux_labels'):
        aux_labels = fsa.aux_labels
        name = 'WFST'
    else:
        aux_labels = None
        name = 'WFSA'

    graph_attr = {
        'rankdir': 'LR',
        'size': '8.5,11',
        'label': '',
        'center': '1',
        'orientation': 'Portrait',
        'ranksep': '0.4',
        'nodesep': '0.25',
    }

    default_node_attr = {
        'shape': 'circle',
        'style': 'bold',
        'fontsize': '14',
    }

    final_state_attr = {
        'shape': 'doublecircle',
        'style': 'bold',
        'fontsize': '14',
    }

    final_state = -1
    dot = Digraph(name=name, graph_attr=graph_attr)

    seen = set()
    i = -1
    for arc, weight in zip(fsa.arcs.values()[:, :-1], fsa.score.tolist()):
        i += 1
        src_state, dst_state, label = arc.tolist()
        src_state = str(src_state)
        dst_state = str(dst_state)
        label = int(label)
        if label == -1:
            final_state = dst_state
        if src_state not in seen:
            dot.node(src_state, label=src_state, **default_node_attr)
            seen.add(src_state)

        if dst_state not in seen:
            if dst_state == final_state:
                dot.node(dst_state, label=dst_state, **final_state_attr)

            else:
                dot.node(dst_state, label=dst_state, **default_node_attr)
            seen.add(dst_state)
        if aux_labels is not None:
            aux_label = int(aux_labels[i])
            if hasattr(fsa, 'osym'):
                aux_label = fsa.osym.get(aux_label)
            aux_label = f':{aux_label}'
        else:
            aux_label = ''

        if hasattr(fsa, 'isym') and label != -1:
            label = fsa.isym.get(label)

        dot.edge(src_state,
                 dst_state,
                 label=f'{label}{aux_label}/{weight:.2f}')
    return dot

# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

import torch  # noqa
import _k2

from .fsa import Fsa


def get_properties(fsa: Fsa) -> int:
    '''Get the properties of an FSA.

    Note that the properties of an FSA is encoded into an integer.
    The integer is expected to be passed to one of the `is_*` functions
    in this module, e.g., :func:`is_arc_sorted`.

    Args:
      fsa:
        The input FSA.

    Returns:
      An integer which encodes the properties of the given FSA.
    '''
    return _k2.get_fsa_basic_properties(fsa.arcs)


def properties_to_str(p: int) -> str:
    '''Convert properties to a string for debug purpose.

    Args:
      p:
        An integer returned by :func:`get_properties`.

    Returns:
      A string representation of the input properties.
    '''
    return _k2.fsa_properties_as_str(p)


def is_arc_sorted(p: int) -> bool:
    '''Determine whether the given properties imply an arc_sorted FSA.

    Args:
      p:
        An integer returned by :func:`get_properties`.

    Returns:
      True if `p` implies an arc_sorted FSA.
      False otherwise.
    '''
    return _k2.is_arc_sorted(p)


def is_accessible(p: int) -> bool:
    '''Determine whether the given properties imply an FSA that
    has all states being accessible.

    Args:
      p:
        An integer returned by :func:`get_properties`.

    Returns:
      True if `p` implies an FSA with all states being accessible.
      False otherwise.
    '''
    return _k2.is_accessible(p)


def is_coaccessible(p: int) -> bool:
    '''Determine whether the given properties imply an FSA that
    has all states being coaccessible.

    Args:
      p:
        An integer returned by :func:`get_properties`.

    Returns:
      True if `p` implies an FSA with all states being coaccessible.
      False otherwise.
    '''
    return _k2.is_coaccessible(p)

# This file is for generating documentation
# and the actual implementation is in
#  k2/k2/python/csrc/torch/ragged.cu

import torch


class RaggedInt:
    pass


class RaggedArc:
    pass


class RaggedShape:

    def __init__(self, src: str):
        '''Construct a ragged shape from a string.

        For example:

            .. code-block:: python

                s = '[ [x x] [x] [x x x] ]'
                shape = k2.RaggedShape(s)
                assert shape.dim0() == 3
                assert shape.num_axes() == 2
                assert shape.max_size(1) == 2
                assert shape.num_elements() == 6
                assert shape.tot_size(1) == 6
                assert torch.all(torch.eq(shape.row_ids(1), torch.tensor([0, 0, 1, 2, 2, 2])))
                assert torch.all(torch.eq(shape.row_splits(1), torch.tensor([0, 2, 3, 6])))

        Args:
          src:
            A string representation of the ragged shape. See its
            usage in the above example code.
        '''

    def dim0(self) -> int:
        '''Returns number of elements of dimension 0.

        For example:

            - If this shape is associated with an FSA, it returns the
              number of states.

            - If this shape is associated with an FsaVec, it returns
              the number of FSAs in the vector.
        '''

    def num_axes(self) -> int:
        '''Returns number of axes.

        For example:

            - If it is an FSA, it returns 2.
            - If it is an FsaVec, it returns 3.
        '''

    def max_size(self, axis: int) -> int:
        '''Gives max size of any list on the provided axis
        with 0 < `axis` < :func:`num_axes`.

        Equals max difference between successive `row_splits` on that axis.

        For example:

            - If this shape is associated with an FSA, set `axis=1` will
              return the maximum number of arcs any state has.

        Args:
          axis: max size of this axis is returned.
        '''

    def num_elements(self) -> int:
        '''Returns the number of elements that a ragged array with
        this shape would have
        '''

    def tot_size(self, axis: int) -> int:
        '''Return the  total size on this axis.

        Requires 0 <= `axis` < :func:`num_axes` and for `axis = 0` the
        returned value is the same as :func:`dim0`.

        Args:
          axis: total size of this axis is returned.
        '''

    def to(self, device: torch.device) -> 'RaggedShape':
        '''Move this shape to a given device.

        If it is already on the given device, itself is returned.
        Otherwise, a copy of this shape is returned.

        Args:
          device:
            An instance of `torch.device`. It supports only cpu
            and cuda devices.

            Caution:
              It does not support a string parameter like `cpu` or `cuda:0`.
              Please use `torch.device('cpu')` or `torch.device('cuda', 0)`.
        Returns:
          An instance of :class:`RaggedShape` on the given device.
        '''

    def row_ids(self, axis: int) -> torch.Tensor:
        '''Return the row IDs for the given `axis`.

        Caution:
          You should NOT modify the returned `torch.Tensor`.

        Args:
          axis:
            Row IDs of this axis is returned.
            Requires 0 < `axis` < :func:`num_axes`.

        Returns:
          Return the row IDs of this axis in `torch.Tensor`.
        '''

    def row_splits(self, axis: int) -> torch.Tensor:
        '''Return the row splits for the given `axis`.

        Caution:
          You should NOT modify the returned `torch.Tensor`.

        Args:
          axis:
            Row splits of this axis is returned.
            Requires 0 < `axis` < :func:`num_axes`.

        Returns:
          Return the row splits of this axis in `torch.Tensor`.
        '''

    def __str__(self) -> str:
        '''Return a string representation of this shape.
        '''


class DenseFsaVec:
    pass


def simple_ragged_index_select():
    pass


def _as_float():
    pass


def _as_int():
    pass


def _fsa_from_str():
    pass


def _fsa_from_tensor():
    pass


def _fsa_to_str():
    pass


def _fsa_to_tensor():
    pass


def create_ragged_shape2():
    pass


def random_ragged_shape():
    pass


def _create_fsa_vec():
    pass


def _is_rand_equivalent():
    pass

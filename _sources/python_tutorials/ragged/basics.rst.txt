Basics
======

A ragged tensor or ragged array in k2 can be used to store the following kinds of data:

  1. A list of lists. Each sub list may contain different number of entries.
     That is, they can have different lengths.

    .. code-block:: python

      a = [ [1, 2], [2, 3, 4], [], [1] ]

  2. A list-of-list of lists.

    .. code-block:: python

      b = [ [[1., 2.5], [2, 3, 4]], [[]], [[10, 20], []] ]

  3. A list-of-list-of-list-of... lists. List can be nested in any number of levels.


.. Note::

  Ragged arrays are the **core** data structures in k2, designed by us
  **independently**. We were later told that TensorFlow was using the same
  ideas (See `tf.ragged <https://www.tensorflow.org/guide/ragged_tensor>`_).

In k2, a ragged tensor contains two parts:

  - a shape, which is of type :class:`k2.RaggedShape`
  - a value, which can be accessed as a **contiguous**
    `PyTorch tensor <https://pytorch.org/docs/stable/tensors.html>`_.

.. hint::

  The **value** is stored contiguously in memory.



.. container:: toggle

    .. container:: header

        .. attention::

          What is the dimension of the **value** as a torch tensor? (Click ▶ to see it)

    It depends on the data type of of the ragged tensor. For instance,

      - if the data type is ``int32_t``, the **value** is accessed as a **1-D** torch tensor with dtype ``torch.int32``.
      - if the data type is ``float``, the **value** is accessed as a **1-D** torch tensor with dtype ``torch.float32``.
      - if the data type is ``double``, the **value** is accessed as a **1-D** torch tensor with dtype ``torch.float64``.

    If the data type is ``k2::Arc``, which has the following definition
    `in C++ <https://github.com/k2-fsa/k2/blob/master/k2/csrc/fsa.h#L31>`_:

      .. code-block:: c++

        struct Arc {
          int32_t src_state;
          int32_t dest_state;
          int32_t label;
          float score;
        };

    the **value** is acessed as a **2-D** torch tensor with dtype ``torch.int32``.
    The **2-D** tensor has 4 columns: the first column contains ``src_state``,
    the second column contains ``dest_state``, the third column contains ``label``,
    and the fourth column contains ``score`` (The float type is **reinterpreted** as
    int type without any conversions).

    There are only 1-D and 2-D **value** tensors in k2 at present.

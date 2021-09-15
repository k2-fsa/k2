Basics
======

In this tutorial, we describe

  - What are ragged tensors?

     - What are the differences between ragged tensors and regular tensors?
     - How to create ragged tensors?

  - Various concepts relevant to ragged tensors, including

     - What is ``RaggedShape``?
     - What is ``row_splits`` ?
     - What is ``row_ids`` ?
     - What is ``dim0`` ?
     - What is ``tot_size`` ?

What are ragged tensors?
------------------------

Before talking about what ragged tensors are, let's look at what non-ragged
tensors, i.e., regular tensors, look like.

  - 2-D regular tensors

    .. literalinclude:: code/basics/regular-tensors.py
       :language: python
       :lines: 8-20

    The shape of the 2-D regular tensor ``a`` is ``(3, 4)``, meaning it has 3
    rows and 4 columns. Each row has **exactly** 4 elements, no more, no less.

  - 3-D regular tensors

    .. literalinclude:: code/basics/regular-tensors.py
       :language: python
       :lines: 24-45

    The shape of the 3-D regular tensor ``b`` is ``(3, 3, 2)``, meaning it has
    3 planes. Each plane has **exactly** 3 rows, no more, no less. Each row has
    **exactly** two entries, no more, no less.

  - N-D regular tensors (N >= 4)

    We assume you know how to create N-D regular tensors.

After looking at what non-ragged tensors look like, let's have a look at ragged
tensors in ``k2``.

  - 2-D ragged tensors

    .. literalinclude:: code/basics/ragged-tensors.py
       :language: python
       :lines: 7-16

    The 2-D ragged tensor ``c`` has 4 rows. However, unlike regular tensors,
    each row in ``c`` can have different number of elements. In this case,

      - Row 0 has 5 entries: ``[1, 2, 3, 6, -5]``
      - Row 1 has 2 entries: ``[0, 1]``
      - Row 2 is empty. It has no entries.
      - Row 3 has only 1 entry: ``[3]``

    .. Hint::

      In ``k2``, we say that ``c`` is a ragged tensor with **two axes**.

  - 3-D ragged tensors

    .. literalinclude:: code/basics/ragged-tensors.py
       :language: python
       :lines: 20-40

    The 3-D ragged tensor ``d`` has 4 planes. Different from regular tensors,
    different planes in a ragged tensor can have different number of rows.
    Moreover, different rows within a plane can also have different number
    of entries.

    .. Hint::

      In ``k2``, we say that ``d`` is a ragged tensor with **three axes**.

  - N-D ragged tensors (N >= 4)

    Having seen how to create 2-D and 3-D ragged tensors, we assume you know how to
    create N-D ragged tensors.

A ragged tensor in ``k2`` has ``N`` (``N >= 2``) axes. Unlike regular tensors,
each axis of a ragged tensor can have different number of elements.

Ragged tensors are **the most important** data structures in ``k2``. FSAs are
represented as ragged tensors. There are also various operations defined on ragged
tensors.

At this point, we assume you know how to create ``N-D`` ragged tensors in ``k2``.
Let us do some exercises to check what you have learned.

Exercise 1
^^^^^^^^^^

.. container:: toggle

    .. container:: header

        .. Note::

          How to create a ragged tensor with 2 axes, satisfying the following
          constraints:

            - It has 3 rows.
            - Row 0 has elements: ``1, 10, -1``
            - Row 1 is empty, i.e., it has no elements.
            - Row 2 has two elements: ``-1.5, 2``

          (Click ▶ to see it)

    .. literalinclude:: code/basics/ragged-tensors.py
       :language: python
       :lines: 43-49

Exercise 2
^^^^^^^^^^

.. container:: toggle

    .. container:: header

        .. Note::

          How to create a ragged tensor with only 1 axis?

          (Click ▶ to see it)

    You **cannot** create a ragged tensor with only 1 axis. Ragged tensors
    in ``k2`` have at least 2 axes.

Concepts about ragged tensors
-----------------------------

A ragged tensor in ``k2`` consists of two parts:

  - ``shape``, which is an instance of :class:`k2.RaggedShape`

    .. Caution::

      It is assumed that a shape  within a ragged tensor in ``k2`` is a constant.
      Once constructed, you are not expected to modify it. Otherwise, unexpected
      things can happen; you will be SAD.

  - ``data``, which is an **array** of type ``T``

    .. Hint::

      ``data`` is stored ``contiguously`` in memory, whose entries have to be
      of the same type ``T``. ``T`` can be either primitive types, such as
      ``int``, ``float``, and ``double`` or can be user defined types. For instance,
      ``data`` in FSAs contains ``arcs``, which is defined in C++
      `as follows <https://github.com/k2-fsa/k2/blob/master/k2/csrc/fsa.h#L31>`_:

      .. code-block:: c++

          struct Arc {
            int32_t src_state;
            int32_t dest_state;
            int32_t label;
            float score;
          }

In the following, we describe what is inside a ``shape`` and how to manipulate
``data``.

Shape
^^^^^

To be done.

data
^^^^

TBD.

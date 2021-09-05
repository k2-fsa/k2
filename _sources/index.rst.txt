.. k2 documentation master file, created by
   sphinx-quickstart on Fri Oct  2 21:03:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

k2
==

.. image:: _static/logo.png
    :alt: k2 logo
    :width: 168px
    :align: center
    :target: https://github.com/k2-fsa/k2

.. HINT::

  We use the lowercase ``k`` in ``k2``. That is, it is ``k2``, not ``K2``.

k2 is able to seamlessly integrate Finite State
Automaton (FSA) and Finite State Transducer (FST) algorithms into
autograd-based machine learning toolkits like PyTorch [#f1]_.
k2 supports CPU as well as CUDA. It can process a batch of FSTs
at the same time.

.. [#f1] Support for TensorFlow will be added in the future.

We have used k2 to compute CTC loss, LF-MMI loss, and to do decoding
including lattice rescoring in the field of automatic speech recognition
(ASR) [#f2]_.

We hope k2 will have many other applications as well.


.. [#f2] You can find ASR recipes in `<https://github.com/k2-fsa/icefall>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation/index
   core_concepts/index
   python_tutorials/index
   python_api/index

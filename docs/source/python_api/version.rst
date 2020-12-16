version
=======

k2 provides a Python API to collect information about the environment
in which k2 was built::

  python3 -m k2.version

Please attach the above output while creating an issue on GitHub.

If you are using PyTorch with k2, please also attach the
environment information about PyTorch which can be obtained by running::

  python3 -m torch.utils.collect_env

Installation
============

We provide the following approaches to install k2. You can choose any of them that
is appropriate for you.


.. toctree::
   :maxdepth: 1

   from_wheels
   from_source
   for_developers
   cuda-cudnn.rst
   faqs

Reporting issues
----------------

If you encounter any errors while using k2 after installation, please
create an issue `on GitHub <https://github.com/k2-fsa/k2/issues/new>`_
and tell us your current environment by posting the output of the following
two commands::

  python3 -m k2.version
  python3 -m torch.utils.collect_env

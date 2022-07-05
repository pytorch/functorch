Install functorch
=================

pip
---

To install functorch via pip, please first install
`PyTorch 1.11 <https://pytorch.org/get-started/locally/>`_
and then run the following command:

::

  pip install functorch

We currently support manylinux, x86 MacOS, and Windows, please reach out on
`GitHub <https://github.com/pytorch/functorch>`_ for other platforms.

.. warning::
    functorch's Linux binaries are compatible with all PyTorch 1.12.0 binaries
    aside from the PyTorch 1.12.0 cu102 binary; functorch will raise an error
    if it is used with an incompatible PyTorch binary. This is due to a bug in
    PyTorch (`pytorch/pytorch#80489 <https://github.com/pytorch/pytorch/issues/80489>`_)
    and will be fixed in the next PyTorch minor release.

    As a workaround, please install functorch from source:
    ``pip install --user git+https://github.com/pytorch/functorch@v0.2.0``

Colab
-----

Please see `this colab for instructions. <https://colab.research.google.com/drive/1GNfb01W_xf8JRu78ZKoNnLqiwcrJrbYG#scrollTo=HJ1srOGeNCGA>`_


Building from source
--------------------

See our `README <https://github.com/pytorch/functorch#installing-functorch-main>`_
for instructions on how to build the functorch main development branch for the
latest and greatest. This requires an installation of the latest PyTorch nightly.

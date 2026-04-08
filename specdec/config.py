"""
Package-level configuration flags for specdec.

Usage
-----
Set flags before creating any objects::

    from specdec import config
    config.show_plots = True

    decomp = EndmemberDecomposition(ds, n_endmembers=3)
    decomp.run()   # will automatically show plots at key steps

Flags
-----
show_plots : bool
    If ``True``, plotting functions are automatically called at key steps
    of the decomposition workflow (e.g. after endmember initialisation).
    Default ``False``.
"""

#: Automatically display plots at key decomposition steps when ``True``.
show_plots: bool = False

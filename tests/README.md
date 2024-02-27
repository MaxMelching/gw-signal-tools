# gw_signal_tools tests

This is a guide to the test folder of the gw_signal_tools package. In
principle, its structure is very simple: there are Python files starting
with "test_", which are tested using the pytest package, and there are Jupyter
notebooks, which contain "visual tests" of the package (i.e. mostly plots).

These files are meant to complement each other, where the Python files contain
more hands-on statements on agreement etc., whereas the Jupyter notebooks
emphasize claims made during the selection of certain thresholds. Some
of these thresholds might feel a little high to really claim "good" agreement,
but more often than not, there are explanations for this. The notebooks are
suitable for assessing whether certain deviations are indeed problematic.

# -- Standard Lib Imports
from __future__ import annotations  # Needed for "if TYPE_CHECKING" block
import warnings
from typing import Optional, Callable, overload, Literal

# -- Third Party Imports
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.types import Series
import astropy.units as u

# -- Local Package Imports
from ..waveform import (
    inner_product,
    norm,
    _INNER_PROD_ARGS,
    WaveformDerivative,
)
from ..types import MatrixWithUnits, FDWFGen, DerivInfoBase
from ..test_utils import allclose_quantity


__doc__ = """
Module that contains functions to calculate numerical derivatives of
gravitational waveforms and also a wrapper to calculate a Fisher matrix.
"""

__all__ = ('num_diff', 'fisher_matrix')


def num_diff(  # TODO: move this to deriv file?
    signal: Series | np.ndarray, h: Optional[float | u.Quantity] = None
) -> Series | np.ndarray:
    """
    Implementation of five-point stencil method for numerical
    differentiation for numpy arrays and instances of GWpy Series class
    (which includes instances of TimeSeries, FrequencySeries). The
    differentiation is carried out with respect to the respective
    quantity that spans `signal.xindex`.

    At the boundary points, less accurate methods like the central,
    forward and backward difference are used.

    Parameters
    ----------
    signal : ~gwpy.types.series.Series or numpy.ndarray
        Input to compute derivative for.
    h : float or ~astropy.units.Quantity, optional, default = None
        Distance between elements of signal. Is computed automatically
        in case signal is a GWpy Series. If None, is assumed to be 1.

    Returns
    -------
    ~gwpy.types.series.Series or ~numpy.ndarray
        Derivative of `signal`.
    """
    if isinstance(signal, Series):
        if h is None:
            h = signal.dx
        elif not isinstance(h, u.Quantity):
            h = u.Quantity(h, signal.xindex.unit)

        if not allclose_quantity(
            h.value, signal.dx.value, atol=0.0, rtol=1e-3
        ):  # pragma: no cover
            warnings.warn('Given `h` does not coincide with `signal.dx`.')
    else:
        # -- Make sure signal is array, we utilize numpy operations
        signal = np.asarray(signal)

        # -- Check if h is set
        if h is None:
            h = 1.0
        else:
            if isinstance(h, u.Quantity):
                signal = u.Quantity(signal, u.dimensionless_unscaled)

    signal_deriv = (
        np.roll(signal, 2)
        - 8.0 * np.roll(signal, 1)
        + 8.0 * np.roll(signal, -1)
        - np.roll(signal, -2)
    )
    signal_deriv /= 12.0 * h

    signal_deriv[0] = (signal[1] - signal[0]) / h  # Forward difference
    signal_deriv[1] = (signal[2] - signal[0]) / (2.0 * h)  # Central difference

    signal_deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * h)  # Central difference
    signal_deriv[-1] = (signal[-1] - signal[-2]) / h  # Backward difference

    return signal_deriv


@overload
def fisher_matrix(
    point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: FDWFGen,
    deriv_routine: str | Callable,
    *,
    return_info: Literal[False] = ...,
    pass_inn_prod_kwargs_to_deriv: bool = ...,
    **deriv_and_inner_prod_kwargs,
) -> MatrixWithUnits: ...  # pragma: no cover - overloads


@overload
def fisher_matrix(
    point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: FDWFGen,
    deriv_routine: str | Callable,
    *,
    return_info: Literal[True],  # = ...,
    pass_inn_prod_kwargs_to_deriv: bool = ...,
    **deriv_and_inner_prod_kwargs,
) -> tuple[
    MatrixWithUnits, dict[str, dict[str, FrequencySeries | DerivInfoBase]]
]: ...  # pragma: no cover - overloads


def fisher_matrix(
    point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: FDWFGen,
    deriv_routine: str | Callable,
    return_info: bool = False,
    pass_inn_prod_kwargs_to_deriv: bool = False,
    **deriv_and_inner_prod_kwargs,
) -> (
    MatrixWithUnits
    | tuple[MatrixWithUnits, dict[str, dict[str, FrequencySeries | DerivInfoBase]]]
):
    r"""
    Compute Fisher matrix at a fixed point. This function is mainly
    intended for internal use by ``~gw_signal_tools.fisher.FisherMatrix``.

    Parameters
    ----------
    point : dict[str, ~astropy.units.Quantity]
        Point in parameter space at which the Fisher matrix is
        evaluated, encoded as key-value pairs representing
        parameter-value pairs. Given as input to :code:`wf_generator`.
    params_to_vary : str or list[str]
        Parameter(s) with respect to which the derivatives will be
        computed, the norms of which constitute the Fisher matrix.
        Must be a key in :code:`point` or one of :code:`'time'`,
        :code:`'phase'`.

        For the latter, analytical derivatives are applied. This is
        possible because they contribute only to a factor
        :math:`\exp(i \cdot phase - i \cdot 2 \pi \cdot f \cdot time)`
        in the waveform generated by `wf_generator`, i.e. they are
        global phase and time shifts.
        Beware that the polarization angle :code:`psi` might be
        degenerate with :code:`phase`, if you are using the complex
        strain combination :math:`h = h_+ + i \, h_{\times}`.

        The last analytical derivative is the one for the luminosity
        distance :math:`D_L`, which enters in waveforms only as an
        amplitude factor :math:`1/D_L`. Note that can only be done
        if the parameter recognized, i.e. if it is called `'distance'`.
    wf_generator : ~gw_signal_tools.types.FDWFGen
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform.get_wf_generator`, which
        generates a suitable function from a few arguments.
    deriv_routine : string or Callable
        Determines the class used for numerical differentiation. The
        only requirement on this argument is that it is accepted by the
        ``~gw_signal_tools.waveform.deriv.WaveformDerivative`` class
        as the `deriv_routine` argument.
    return_info : boolean, optional, default = True
        Whether to return information collected during the derivative
        calculations. Can be used as a sort of custom cache to also
        return derivatives.
    pass_inn_prod_kwargs_to_deriv : bool, optional, default = False
        If True, all keyword arguments in `deriv_and_inner_prod_kwargs`
        are passed on to the derivative routine as well. Otherwise, only
        those keyword arguments that are not recognized as inner product
        arguments are passed on. This might be useful if the routine
        uses, e.g., a convergence scheme that is based on inner product
        values. Should be passed, e.g., when using
        ``deriv_routine='gw_signal_tools'``.
    deriv_and_inner_prod_kwargs :
        All other keyword arguments are automatically separated and then
        passed to the derivative and inner product routines involved in
        the Fisher matrix calculations. Allowed keywords are those
        accepted by ``~gw_signal_tools.waveform.Derivative`` and by
        ``~gw_signal_tools.waveform.inner_product``.

    Returns
    -------
    ~gw_signal_tools.matrix_with_units.MatrixWithUnits or tuple[~gw_signal_tools.matrix_with_units.MatrixWithUnits, dict[str, dict[str, ~gwpy.frequencyseries.FrequencySeries | ~gw_signal_tools.types.DerivInfoBase]]]
        A ``MatrixWithUnits`` instance. Entries are Fisher values, where
        index :math:`(i, j)` corresponds to the inner product of
        derivatives with respect to the parameters
        :code:`params_to_vary[i]`, :code:`params_to_vary[j]`.

        If `return_info` is True, a tuple is returned. The first entry
        is the Fisher matrix as described above, while the second entry
        is a dictionary that contains, for each parameter in
        :code:`params_to_vary`, a dictionary with the waveform derivative
        and the :code:`info` property collected during its calculation.

    Notes
    -----
    The main reason behind choosing ``MatrixWithUnits`` as the data
    type was that information about units is available from our
    calculations, so simply discarding it would not make sense.
    Moreover, "regular" calculations using e.g. numpy arrays can also
    be carried out fairly easily using this type, namely by extracting
    this value using by applying `.value` to the class instance.

    See Also
    --------
    gw_signal_tools.waveform :
        Contains the classes used for numerical differentiation (names
        are listed above, see the description of `deriv_routine`).
    """
    # -- Separate deriv and inner_prod kwargs, check defaults
    _deriv_kw_args = {}
    _inner_prod_kwargs = {}
    for key, value in deriv_and_inner_prod_kwargs.items():
        if key in _INNER_PROD_ARGS:
            _inner_prod_kwargs[key] = value
        else:
            _deriv_kw_args[key] = value
    _inner_prod_kwargs['return_opt_info'] = False
    # -- Ensures float output of inner_product

    if isinstance(params_to_vary, str):
        params_to_vary = [params_to_vary]

    param_numb = len(params_to_vary)

    # -- Initialize Fisher Matrix as MatrixWithUnits instance
    fisher_matrix = MatrixWithUnits(
        np.zeros(2 * (param_numb,), dtype=float),
        np.full(2 * (param_numb,), u.dimensionless_unscaled, dtype=object),
    )

    # -- Compute relevant derivatives in frequency domain
    deriv_info = {}

    for i, param in enumerate(params_to_vary):
        full_deriv = WaveformDerivative(
            point=point,
            param_to_vary=param,
            wf_generator=wf_generator,
            deriv_routine=deriv_routine,
            **(
                deriv_and_inner_prod_kwargs
                if pass_inn_prod_kwargs_to_deriv
                else _deriv_kw_args
            ),
        )

        deriv, info = full_deriv.deriv, full_deriv.info  # type: ignore[attr-defined]
        if deriv_routine == 'gw_signal_tools':
            fisher_matrix[i, i] = info.norm_squared
        else:
            fisher_matrix[i, i] = norm(deriv, **_inner_prod_kwargs) ** 2  # type: ignore[operator]

        deriv_info[param] = dict(deriv=deriv, info=info)

    # -- Populate remaining entries of Fisher matrix
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):
            if i == j:
                # -- Was already set in previous loop
                continue
            else:
                fisher_matrix[i, j] = fisher_matrix[j, i] = inner_product(
                    deriv_info[param_i]['deriv'],
                    deriv_info[param_j]['deriv'],
                    **_inner_prod_kwargs,
                )

    if return_info:
        return fisher_matrix, deriv_info
    else:
        return fisher_matrix

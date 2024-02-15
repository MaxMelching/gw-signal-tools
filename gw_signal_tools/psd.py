import numpy as np

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u


def psd_to_file(
    psd: FrequencySeries,
    fname: str,
    is_asd: bool = False
) -> None:
    """
    Save power spectral density (PSD) values from a GWPy
    ``FrequencySeries`` into a file.

    Parameters
    ----------
    psd : ~gwpy.frequencyseries.FrequencySeries
        Values to save.
    fname : str
        File to write PSDs and Frequencies to.
    is_asd : bool, optional, default = False
        If true, values in `psd` are taken to be ASD values rather than
        PSD values and thus a squared version of them is saved.

    Returns
    -------
    None

    See also
    --------
    numpy.savetxt : Used to read the file.
    """

    if is_asd:
        psd = psd**2

    np.savetxt(fname, np.transpose([psd.frequencies.value, psd.value]))


def psd_from_file(
    fname: str,
    is_asd: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Power spectral density (PSD) values from a file into numpy
    arrays. The file must be readable by `~numpy.loadtxt`.

    Parameters
    ----------
    fname : str
        File with two columns, left one representing frequencies and
        right one representing the corresponding PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.

    Returns
    -------
    freqs, psd : tuple[~numpy.array, ~numpy.array]
        Frequencies and PSD values as numpy arrays.

    See also
    --------
    numpy.loadtxt : Used to read the file.
    """

    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    return freqs, psd


def psd_from_file_to_FreqSeries(
    fname: str,
    is_asd: bool = False,
    **kwargs
) -> FrequencySeries:
    """
    Read Power spectral density (PSD) values from file into a GWpy
    ``FrequencySeries``.

    Parameters
    ----------
    fname : str
        File with two columns, left one representing frequencies and
        right one representing the corresponding PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.
    **kwargs
        Other keyword arguments that are passed to ``FrequencySeries``
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries
        PSD as a ``FrequencySeries``.
    """

    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    freqs, psd = psd_from_file(fname, is_asd=is_asd)

    return FrequencySeries(
        psd,
        frequencies=freqs,
        unit=1 / u.Hz,  # TODO: change to strain/Hz once lal updates are incorporated
        **kwargs
    )


def get_FreqSeries_from_dict(
    psd: dict,
    psd_vals_key: str,
    is_asd: bool = False,
    **kwargs
) -> FrequencySeries:
    """
    Converts dictionary with Power spectral density (PSD) values into a
    GWpy ``FrequencySeries``. Frequencies are expected to be accessible
    using the key 'frequencies'.

    Parameters
    ----------
    psd : dict
        Dictionary with PSD values and corresponding frequencies.
    psd_vals_key : str
        Key that holds PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.
    **kwargs
        Other keyword arguments that are passed to ``FrequencySeries``
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries
        Data from input dict in a ``FrequencySeries``.
    """

    return FrequencySeries(
        psd[psd_vals_key]**2 if is_asd else psd[psd_vals_key],
        frequencies=psd['frequencies'],
        **kwargs
    )
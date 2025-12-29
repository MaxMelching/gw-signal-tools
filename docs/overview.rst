Code Overview
=============

Here we talk about which features are implemented in gw-signal-tools. The
focus is a high-level overview, without going in too much detail about
implementations etc., which is done in the examples section.


:code:`inner_product` Module
----------------------------

Calculates the noise-weighted inner product

.. math::
    \langle a, b \rangle = 2 \Re \int_{-\infty}^{\infty}
    \frac{\tilde{a}(f) \tilde{b}^*(f)}{S_n(f)} \, df

of two signals using their representations
:math:`\tilde{a}(f), \tilde{b}(f)` in frequency domain.

Can be done either by evaluating integral using Simpson rule or by
using formulation in terms of inverse Fourier transform. This may be
convenient in case optimization over time and phase shifts between
signals is supposed to be carried out.


:code:`fisher` Module
---------------------

FisherMatrix allows to calculate the statistical error

.. math::
    \Delta \theta^\mu_\mathrm{stat} = \sqrt{\Gamma^{-1}_{\mu \mu}}

and systematic error

.. math::
    \Delta \theta^\mu_\mathrm{stat} = \sum_{\nu} \Gamma^{-1}_{\mu \nu}
    \langle \frac{\partial h}{\partial \theta^\nu}, \delta h \rangle

where :math:`\delta h = h_\mathrm{ref} - h_\mathrm{approx}`.


TODO: improve this!!

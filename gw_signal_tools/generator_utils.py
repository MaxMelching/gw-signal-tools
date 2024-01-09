import astropy.units as u
from lalsimulation.gwsignal.core.utils import add_params_units, check_dict_parameters
from lalsimulation.gwsignal.core.parameter_conventions import SI_units_dictionary


__doc__ = """
Helper functions for waveform generation with lal.
"""



def make_params_dict(
    **kwargs
) -> dict:
    
    # default_dict = {
    #     # Intrinsic parameters
    #     'mass1': 1.0 * u.solMass,
    #     'mass2': 1.0 * u.solMass,
    #     'spin1': 0.0,  # TODO: correct; need to give x, y, z of this
    #     'spin2': 0.0,
    #     'eccentricity': 0.0 * u.dimensionless_unscaled,

    #     # Extrinsic parameters
    #     'distance': 1.0 * u.Mpc,
    #     'inclination': 0.0 * u.rad,
        
    #     # Sampling parameters
    #     'deltaT': 1.0 / 64.0 * u.s,
    #     'deltaF': 1.0 / 16 * u.Hz,
    #     'f22_start': 20.0 * u.Hz,
    #     'f22_ref': 20.0 * u.Hz,  # Put to intrinsic?

    #     # TODO: check if following ones are necessary. If yes, look what they are
    #     'longAscNodes': 0.0 * u.rad,
    #     'meanPerAno': 0.0 * u.rad,
    #     'phiRef': 0.0 * u.rad,
    #     'condition': 0.0
    # }

    default_dict = SI_units_dictionary


    # Type checking input
    # for parameter, val in kwargs.items():
    #     if not isinstance(val, u.Quantity):
    #         # TODO: handle condition keyword argument if it is needed
    #         print((f'Parameter {parameter} has a value of {val}, which has no'
    #                f'astropy unit. This may cause trouble in lal functions.'))

    #         kwargs[parameter] = u.Quantity(val)

    # kwargs = add_params_units(kwargs)  # Adds units
    # Hmmm, this one adds SI units (makes sense, but wrong scale)... So maybe rather do following
    for parameter, val in kwargs.items():
        if not isinstance(val, u.Quantity):
            print(f'Parameter {parameter} has a value of {val}, which has no '
                  'astropy unit. The corresponding SI unit will be added, but '
                  'beware of an eventual scaling that you intended to apply.')
            
            # TODO: built in option for cosmological units? They have solarMass

            # kwargs[parameter] = u.Quantity(val)
            # kwargs = add_params_units(kwargs)
            # break  # All are checked if a single check is performed

            # Or maybe do:
            kwargs |= add_params_units({parameter: val})
            # This will only add unit to this one parameter

            # Or like this:
            kwargs[parameter] = u.Quantity(val, unit=SI_units_dictionary[parameter])
            # TODO: change this to units_dict[unit_sys]?
            # TODO: accept 'SI', i.e. automatically change to 'S.I.'

    # kwargs = add_params_units(kwargs)  # Hmmm, changes values to SI... Desired?
    # check_dict_parameters(kwargs)
    
    # TODO: maybe handle aliases for certain parameters? For example
    # f_min for f22_start or f_ref for f22_ref? I.e. look if these are
    # in kw_args and replace respective correctly named parameter
    # -> will we need this in case chi_eff etc are given?
    

    # default_dict |= kwargs
    # This operation replaces default values if corresponding keyword
    # argument is given

    default_dict = kwargs

#     for kw in kw_args:
#         default_dict = 
    
    return default_dict


"""
Thoughts on param function:

- in principle, many things are implemented somewhere in gwsignal, which is nice

- however, not everything works as we would like to for personal calculations

- therefore, maybe make function that sets stuff for our purposes? For example
  units to solMass and Mpc by default etc.
  
  Could achieve this via default_dict |= personal_default_dict where latter one
  contains only values of stuff that we want to change

  Could also change personal_default_dict beforehand according to given input,
  for example set our default units in case None are given. Or make custom
  warning as it was done above. And set custom f_max maybe.
  -> although this could be handled via kwargs, i.e. optionally
"""


# Define one for GW150914? Because this is always used as reference

PARAMETERS_GW150914 = make_params_dict(
    mass1=36.0 * u.solMass,
    mass2=29.0 * u.solMass,
    distance=440.0 * u.Mpc,
    inclination = 2.7*u.rad,  # Value taken from posteriors.ipynb, where posterior of inclination is plotted
)



# Copying what is done in CompactBinaryCoalescenceGenerator
# -> not entirely sure if this is carried out each time, though
from lalsimulation.gwsignal.core import utils as ut
from lalsimulation.gwsignal.core import parameter_conventions as pc

def parameter_check(self, units_sys='S.I.', extra_parameters=dict(), **parameters):
    """
    Perform checks on the various parameters and populate the different parameters not passed
    in the kwargs required for generating waveforms.

    Parameters
    ----------

        Python dictionary of parameters required for waveform generation
        of the form specified in `parameter_conventions.py`.

    Returns
    -------

        Populate self.waveform_dict with python dictionary and self.lal_dict with LALSuite dictionary structure.
    """
    default_dict = pc.default_dict.copy()
    # Need to add this line to take care of the extra parameters passed to ExternalPython LAL Generator
    ExternalPythonParameters=['object', 'module']

    for key, value in parameters.items():
        if key in ExternalPythonParameters:
            pass
        else:
            default_dict[key] = value

    if not 'deltaF' in default_dict:
        default_dict['deltaF'] = 1./16.*u.Hz
    if not 'deltaT' in default_dict:
        default_dict['deltaT'] = 1./512.*u.s
    if not 'f_max' in default_dict:
        default_dict['f_max'] = (0.5/default_dict['deltaT'].value)*u.Hz

    # Add units if indicated
    if units_sys is not None:
        self.waveform_dict = ut.add_params_units(default_dict, units_sys, generic_param_dict=extra_parameters)
    #This is a mandatory check that units are correctly used
    ut.check_dict_parameters(default_dict, generic_param_dict=extra_parameters)

    self.lal_dict  = ut.to_lal_dict(default_dict)
    return default_dict




# Notes from docs of gwsignal:
# - seems like condition has to be 1 (i.e. True), otherwise error

print(__name__)
if __name__ == '__main__':
    # print(PARAMETERS_GW150914)

    # print(make_params_dict(mass1=30))

    print(make_params_dict(mass1=30 * u.solMass))
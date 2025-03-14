from . import unit_sys as gw_signal_tools_units

preferred_unit_system = gw_signal_tools_units
# NOTE: definition of this quantity is reason why units is a folder/
# module and not a single file. We use this variable throughout the
# whole package and also want to be able to set it. Thus we need a
# single point for definition (instead of importing module each time)


__doc__ = gw_signal_tools_units.__doc__

__all__ = ('preferred_unit_system',)

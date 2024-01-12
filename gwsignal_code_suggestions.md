# Code Suggestions for gwsignal

## Parameter Handling

Change handling of parameters because right now, waveform generation using
masses other than m1, m2 does not work.

-> possible fix: remove `mass1` and `mass2` from `core/parameter_conventions.py/default_dict`
and then change order in l. 353ff of `core/waveform.py`; problem is that
`self.parameter_check` adds keys for masses and spins before it is checked
if they are already determined by input; so perhaps remove these from the
default_dict and first check determination; only if this gives error we add
default values (although one has to distinguish between nothing given or
just wrong determination)

-> maybe one could even use something like `core/utils.py/check_dict_parameters`

-> proposed way of handling errors that are potentially raised by
`CheckDeterminationOfMasses` and `CheckDeterminationOfSpins`:
do try... and catch errors, then print message that no appropriate
combination of masses was given, so default values will be set?
Would then have to remove mass parameters (popping all from mass_dict) and add some default ones
(e.g. the ones that are now in default_dict)

-> intended dictionary difference (other way than popping):

```python
filtered_wf_dict = {key: wf_dict[key] for key in (set(wf_dict) - set(masses_dict))}
```

test of this:

```python
dict1 = {'one': 1, 'two': 2, 'three': 3}
dict2 = {'two': 2, 'four': 4}

dict_12 = {key: dict1[key] for key in set(dict1) - set(dict2)}

print(dict_12)

>>> {'one': 1, 'three': 3}
```

Maybe one can even copy error message and then say "check function failed
with error message ... Proceeding with masses of m1=1.0, m2=1.0".

This gives information in case one intended to set correct combination,
but is also useful in case one just wants default values to be set.

-> I think it is good idea in general to give information about automatic
setting

-> note: my temporary fix was removing l. 174 in `core/waveform.py`, where
`default_dict` is used in `parameter_check`, replaced with empty dictionary

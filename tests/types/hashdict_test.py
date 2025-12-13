# -- Local Package Imports
from gw_signal_tools.types import HashableDict


def test_hashing():
    hash_dict = HashableDict(a=2)

    hash_dict.__hash__()
    hash(hash_dict)


def test_union():
    dict_no_hash = dict(a=1)
    dict_hash = HashableDict(a=2)

    left_merge = dict_hash | dict_no_hash
    assert (
        isinstance(left_merge, HashableDict)  # Check correct type
        and left_merge['a'] == 1  # Check correct merge
    )  # or method

    right_merge = dict_no_hash | dict_hash
    assert (
        isinstance(right_merge, HashableDict)  # Check correct type
        and right_merge['a'] == 2  # Check correct merge
    )  # ror method

    assert not isinstance(dict_no_hash, HashableDict)
    # -- Ensure that dict is not recognized as HashableDict, would mean
    # -- previous results are meaningless

    assert not isinstance(dict_no_hash | dict_no_hash, HashableDict)
    # -- Ensure that interplay of regular dictionaries is unaffected


def test_copy():
    # -- Make sure that copy is still a HashableDict
    dict_hash = HashableDict(a=2)
    dict_hash_copy = dict_hash.copy()
    dict_hash_copy['a'] = 1

    hash(dict_hash_copy)

    assert dict_hash['a'] == 2  # Also check that copying itself works

    hash(dict_hash.__copy__())

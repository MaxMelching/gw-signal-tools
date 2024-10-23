def test_enable_disable_calls():
    from gw_signal_tools.caching import (
        use_caching, cache_func, lru_cache, _dummy_cache, enable_caching,
        disable_caching
    )

    # -- Check defaults
    assert use_caching == False
    assert cache_func == _dummy_cache

    # -- Check enabling
    enable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert use_caching == True
    assert cache_func != _dummy_cache
    # -- Testing for equality with lru_cache does not work, is
    # -- different function "instance" than the one in caching file

    # -- Check disabling
    disable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert use_caching == False
    assert cache_func == _dummy_cache


def test_context_enabling():
    from gw_signal_tools.caching import (
        enable_caching, disable_caching, enable_caching_locally
    )

    # -- Make sure caching stays disabled if it was
    disable_caching()

    from gw_signal_tools.caching import use_caching
    assert use_caching == False

    with enable_caching_locally():
        from gw_signal_tools.caching import use_caching
        assert use_caching == True
    
    from gw_signal_tools.caching import use_caching
    assert use_caching == False

    # -- Make sure caching stays enabled if it was
    enable_caching()

    from gw_signal_tools.caching import use_caching
    assert use_caching == True

    with enable_caching_locally():
        from gw_signal_tools.caching import use_caching
        assert use_caching == True
    
    from gw_signal_tools.caching import use_caching
    assert use_caching == True


def test_context_disabling():
    from gw_signal_tools.caching import (
        enable_caching, disable_caching, disable_caching_locally
    )

    # -- Make sure caching stays enabled if it was
    enable_caching()

    from gw_signal_tools.caching import use_caching
    assert use_caching == True

    with disable_caching_locally():
        from gw_signal_tools.caching import use_caching
        assert use_caching == False
    
    from gw_signal_tools.caching import use_caching
    assert use_caching == True

    # -- Make sure caching stays disabled if it was
    disable_caching()

    from gw_signal_tools.caching import use_caching
    assert use_caching == False

    with disable_caching_locally():
        from gw_signal_tools.caching import use_caching
        assert use_caching == False
    
    from gw_signal_tools.caching import use_caching
    assert use_caching == False

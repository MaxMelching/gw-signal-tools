def test_enable_disable_calls():
    from gw_signal_tools.caching import (
        use_caching, cache_func, cache, _dummy_cache, enable_caching,
        disable_caching
    )

    # -- Check defaults
    assert use_caching == False
    assert cache_func == _dummy_cache

    # -- Check enabling
    enable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert use_caching == True
    assert cache_func == cache

    # -- Check disabling
    disable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert use_caching == False
    assert cache_func == _dummy_cache

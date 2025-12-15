def test_enable_disable_calls():
    from gw_signal_tools.caching import (
        use_caching,
        cache_func,
        _dummy_cache,
        enable_caching,
        disable_caching,
    )

    # -- Check defaults
    assert not use_caching
    assert cache_func == _dummy_cache

    # -- Check enabling
    enable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert use_caching
    assert cache_func != _dummy_cache
    # -- Testing for equality with lru_cache does not work, is
    # -- different function "instance" than the one in caching file

    # -- Check disabling
    disable_caching()
    from gw_signal_tools.caching import use_caching, cache_func

    assert not use_caching
    assert cache_func == _dummy_cache


def test_context_enabling():
    from gw_signal_tools.caching import (
        enable_caching,
        disable_caching,
        enable_caching_locally,
    )

    # -- Make sure caching stays disabled if it was
    disable_caching()

    from gw_signal_tools.caching import use_caching

    assert not use_caching

    with enable_caching_locally():
        from gw_signal_tools.caching import use_caching

        assert use_caching

    from gw_signal_tools.caching import use_caching

    assert not use_caching

    # -- Make sure caching stays enabled if it was
    enable_caching()

    from gw_signal_tools.caching import use_caching

    assert use_caching

    with enable_caching_locally():
        from gw_signal_tools.caching import use_caching

        assert use_caching

    from gw_signal_tools.caching import use_caching

    assert use_caching


def test_context_disabling():
    from gw_signal_tools.caching import (
        enable_caching,
        disable_caching,
        disable_caching_locally,
    )

    # -- Make sure caching stays enabled if it was
    enable_caching()

    from gw_signal_tools.caching import use_caching

    assert use_caching

    with disable_caching_locally():
        from gw_signal_tools.caching import use_caching

        assert not use_caching

    from gw_signal_tools.caching import use_caching

    assert use_caching

    # -- Make sure caching stays disabled if it was
    disable_caching()

    from gw_signal_tools.caching import use_caching

    assert not use_caching

    with disable_caching_locally():
        from gw_signal_tools.caching import use_caching

        assert not use_caching

    from gw_signal_tools.caching import use_caching

    assert not use_caching

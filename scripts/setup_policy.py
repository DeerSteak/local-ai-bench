"""Pure setup-policy decisions shared with the test suite."""


def should_install_llamacpp(os_name: str, llamacpp_found: bool) -> bool:
    """Linux always refreshes its repo-local source build."""
    return os_name == "Linux" or not llamacpp_found

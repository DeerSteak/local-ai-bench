from setup_policy import should_install_llamacpp


def test_linux_refreshes_llamacpp_when_system_binary_exists():
    assert should_install_llamacpp("Linux", llamacpp_found=True) is True


def test_linux_installs_llamacpp_when_missing():
    assert should_install_llamacpp("Linux", llamacpp_found=False) is True


def test_other_platforms_keep_existing_llamacpp():
    assert should_install_llamacpp("Darwin", llamacpp_found=True) is False
    assert should_install_llamacpp("Windows", llamacpp_found=True) is False


def test_other_platforms_install_llamacpp_when_missing():
    assert should_install_llamacpp("Darwin", llamacpp_found=False) is True
    assert should_install_llamacpp("Windows", llamacpp_found=False) is True

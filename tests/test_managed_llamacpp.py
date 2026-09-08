from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.setup import managed_llamacpp as managed
from scripts.setup.runtime_update import RuntimeUpdateResult


def release(*names):
    return {"tag_name": "b10809", "assets": [
        {"name": f"llama-b10809-bin-{name}", "browser_download_url": 'https://example.test/asset', "size": 10}
        for name in names]}


@pytest.mark.parametrize('identity,expected', [
    ({'ID': 'ubuntu'}, True), ({'ID': 'linuxmint', 'ID_LIKE': 'ubuntu debian'}, True),
    ({'ID': 'kubuntu', 'ID_LIKE': 'ubuntu'}, True),
    ({'ID': 'fedora', 'ID_LIKE': 'rhel'}, False), ({'ID': 'rhel'}, False),
    ({'ID': 'arch'}, False), ({'ID': 'debian'}, False),
    ({'ID': 'rex', 'ID_LIKE': 'debian'}, False), ({'ID_LIKE': 'notubuntu'}, False), ({}, False),
])
def test_ubuntu_download_eligibility(identity, expected):
    assert managed.ubuntu_compatible(identity) is expected


@pytest.mark.parametrize('system,machine,backend,asset,kwargs', [
    ('Darwin', 'arm64', 'metal', 'macos-arm64.tar.gz', {}),
    ('Darwin', 'x86_64', 'metal', 'macos-x64.tar.gz', {}),
    ('Linux', 'x86_64', 'cpu', 'ubuntu-x64.tar.gz', {}),
    ('Linux', 'aarch64', 'cpu', 'ubuntu-arm64.tar.gz', {}),
    ('Linux', 'x86_64', 'vulkan', 'ubuntu-vulkan-x64.tar.gz', {}),
    ('Linux', 'aarch64', 'vulkan', 'ubuntu-vulkan-arm64.tar.gz', {}),
    ('Linux', 'x86_64', 'xpu', 'ubuntu-sycl-fp32-x64.tar.gz', {}),
    ('Linux', 'x86_64', 'rocm', 'ubuntu-rocm-10.0-x64.tar.gz', {'rocm_version': (10, 0)}),
    ('Windows', 'AMD64', 'vulkan', 'win-vulkan-x64.zip', {}),
    ('Windows', 'AMD64', 'xpu', 'win-sycl-x64.zip', {}),
    ('Windows', 'arm64', 'cpu', 'win-arm64.zip', {}),
])
def test_asset_selection_matches_os_architecture_and_backend(system, machine, backend, asset, kwargs):
    data = release(asset, 'ubuntu-x64.tar.gz', 'win-vulkan-x64.zip')
    selected = managed.release_assets(data, system, machine, backend, os_release={'ID': 'ubuntu'}, **kwargs)
    assert selected[0]['name'].endswith(asset)


@pytest.mark.parametrize('backend,version', [('cuda', None), ('rocm', None), ('rocm', (7, 2))])
def test_gpu_download_never_uses_cpu_or_incompatible_rocm(backend, version):
    assert not managed.release_assets(release('ubuntu-x64.tar.gz', 'ubuntu-rocm-10.0-x64.tar.gz'),
        'Linux', 'x86_64', backend, rocm_version=version, os_release={'ID': 'ubuntu'})


def test_windows_cuda_requires_driver_compatible_binary_and_runtime():
    data = release('win-cuda-12.4-x64.zip')
    data['assets'].append({'name': 'cudart-llama-bin-win-cuda-12.4-x64.zip'})
    selected = managed.release_assets(data, 'Windows', 'AMD64', 'cuda', max_cuda_version='12.8')
    assert len(selected) == 2
    assert all(isinstance(asset, dict) for asset in selected)
    assert not managed.release_assets(data, 'Windows', 'AMD64', 'cuda', max_cuda_version='12.0')


@pytest.fixture
def installation(tmp_path, monkeypatch):
    target = tmp_path / 'llama.cpp'
    target.mkdir()
    (target / 'old').touch()
    calls = []
    state = {'archive_valid': True, 'build_success': True, 'final_valid': True}
    def download(url, path, **kwargs):
        calls.append('download')
        return path
    def extract(archive, destination):
        destination.mkdir(parents=True)
        (destination / 'binary').touch()
    def validate(path, **kwargs):
        assert kwargs['required_backend'] == 'cpu'
        valid = state['final_valid'] if path == target else (path / 'source').exists() or state['archive_valid']
        return RuntimeUpdateResult(valid, 'validation', '10809')
    def build(path, backend, *, release_fetcher, **kwargs):
        calls.append(('build', release_fetcher()['tag_name'], kwargs['os_name']))
        (path / 'source').touch()
        return RuntimeUpdateResult(state['build_success'], 'source build', '10809')
    monkeypatch.setattr(managed, 'safe_extract_tar', extract)
    monkeypatch.setattr(managed, 'safe_extract_zip', extract)
    def install(system='Linux', identity=None, assets=True, control=None):
        return managed.install_managed_llamacpp(target, system, 'x86_64', 'cpu',
            os_release=identity if identity is not None else {'ID': 'ubuntu'},
            release_fetcher=lambda: release('ubuntu-x64.tar.gz', 'win-x64.zip') if assets else release(),
            downloader=download, validator=validate, source_builder=build, control=control, log=lambda _:None)
    return SimpleNamespace(target=target, calls=calls, state=state, install=install)


def test_compatible_archive_replaces_old_runtime_without_compiling(installation):
    assert installation.install().success
    assert installation.calls == ['download']
    assert (installation.target / 'binary').exists()
    assert not (installation.target / 'old').exists()


@pytest.mark.parametrize('identity', [{'ID': 'fedora'}, {'ID': 'rhel'}, {'ID': 'arch'}, {'ID': 'rex', 'ID_LIKE': 'debian'}])
def test_other_linux_distributions_build_without_downloading(installation, identity):
    assert installation.install(identity=identity).success
    assert installation.calls == [('build', 'b10809', 'posix')]


@pytest.mark.parametrize('system', ['Linux', 'Windows', 'Darwin'])
def test_missing_archive_builds_the_selected_release(installation, system):
    assert installation.install(system=system, assets=False).success
    assert installation.calls == [('build', 'b10809', 'nt' if system == 'Windows' else 'posix')]


def test_incompatible_archive_falls_back_to_same_release(installation):
    installation.state['archive_valid'] = False
    assert installation.install().success
    assert installation.calls == ['download', ('build', 'b10809', 'posix')]
    assert (installation.target / 'source').exists()
    assert not (installation.target / 'binary').exists()


def test_failed_build_preserves_previous_runtime(installation):
    installation.state.update(archive_valid=False, build_success=False)
    assert not installation.install().success
    assert (installation.target / 'old').exists()


def test_final_path_validation_failure_rolls_back(installation):
    installation.state['final_valid'] = False
    assert not installation.install().success
    assert (installation.target / 'old').exists()


def test_cancelled_download_does_not_build_or_replace(installation):
    control = SimpleNamespace(cancelled=True, run=lambda *a, **k: None)
    assert not installation.install(control=control).success
    assert not installation.calls
    assert not any(isinstance(call, tuple) for call in installation.calls)
    assert (installation.target / 'old').exists()


def test_fresh_install_does_not_need_an_existing_runtime(installation):
    import shutil
    shutil.rmtree(installation.target)
    assert installation.install().success
    assert (installation.target / 'binary').exists()


def test_broken_archive_uses_source_without_touching_old_runtime(installation, monkeypatch):
    def extract(*args):
        assert (installation.target / 'old').exists()
        raise ValueError('invalid archive')
    monkeypatch.setattr(managed, 'safe_extract_tar', extract)
    assert installation.install().success
    assert installation.calls == ['download', ('build', 'b10809', 'posix')]


def test_source_only_path_does_not_download_on_unknown_distribution(installation):
    assert installation.install(identity={}).success
    assert installation.calls == [('build', 'b10809', 'posix')]


@pytest.mark.parametrize('system,machine', [('Linux', 'riscv64'), ('Windows', 'arm64'), ('Darwin', 'riscv64')])
def test_architecture_mismatch_never_selects_x64_gpu_asset(system, machine):
    assert not managed.release_assets(release('ubuntu-vulkan-x64.tar.gz', 'win-vulkan-x64.zip',
                                              'macos-x64.tar.gz'), system, machine, 'vulkan',
                                      os_release={'ID': 'ubuntu'})


@pytest.mark.parametrize('installed,assets,expected', [
    ((7, 2), ['ubuntu-rocm-10.0-x64.tar.gz'], True),
    ((9, 9), ['ubuntu-rocm-10.0-x64.tar.gz'], True),
    ((10, 0), ['ubuntu-rocm-10.0-x64.tar.gz'], False),
    ((10, 1), ['ubuntu-rocm-10.0-x64.tar.gz'], False),
    ((7, 2), ['ubuntu-rocm-7.2-x64.tar.gz', 'ubuntu-rocm-10.0-x64.tar.gz'], False),
    ((7, 2), ['ubuntu-rocm-10.0-arm64.tar.gz'], False),
    (None, ['ubuntu-rocm-10.0-x64.tar.gz'], False),
    ((7, 2), ['ubuntu-rocm-invalid-x64.tar.gz'], False),
])
def test_rocm_archive_warning_uses_actual_release_requirements(installed, assets, expected):
    warning = managed.rocm_archive_warning(release(*assets), 'Linux', 'x86_64', 'rocm',
                                           installed, {'ID':'linuxmint', 'ID_LIKE':'ubuntu debian'})
    assert bool(warning) is expected
    if warning:
        assert 'ROCm 10.0' in warning
        assert 'Building against your installed ROCm' in warning
        assert 'will not upgrade ROCm automatically' in warning


@pytest.mark.parametrize('system,backend,identity', [
    ('Linux', 'rocm', {'ID':'rex', 'ID_LIKE':'debian'}),
    ('Linux', 'cuda', {'ID':'ubuntu'}), ('Windows', 'rocm', {'ID':'ubuntu'}),
])
def test_rocm_archive_warning_does_not_apply_to_other_platforms(system, backend, identity):
    assert managed.rocm_archive_warning(release('ubuntu-rocm-10.0-x64.tar.gz'),
        system, 'x86_64', backend, (7, 2), identity) is None


def test_old_rocm_warning_precedes_source_build_without_downloading(tmp_path):
    events = []
    def build(*a, **k):
        events.append('build')
        return RuntimeUpdateResult(False, 'mock build stopped')
    result = managed.install_managed_llamacpp(tmp_path/'runtime', 'Linux', 'x86_64', 'rocm',
        os_release={'ID':'ubuntu'}, rocm_version=(7, 2),
        release_fetcher=lambda: release('ubuntu-rocm-10.0-x64.tar.gz'),
        warn=lambda message: events.append(message), log=lambda _:None, source_builder=build,
        downloader=lambda *a, **k: pytest.fail('incompatible archive must not download'))
    assert not result.success
    assert 'Installed ROCm 7.2' in events[0]
    assert events[1] == 'build'

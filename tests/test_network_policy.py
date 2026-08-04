import os
import socket

import pytest

from scripts.runtime.network_policy import OFFLINE_ENVIRONMENT, OfflineNetworkError, is_loopback_address


@pytest.mark.parametrize("address", [
    ("127.0.0.1", 8080), ("::1", 8080), ("localhost", 8080), "/tmp/local.sock",
])
def test_loopback_policy_allows_local_services(address):
    assert is_loopback_address(address)


@pytest.mark.parametrize("address", [
    ("8.8.8.8", 53), ("example.com", 443), ("192.168.1.2", 8188), None,
])
def test_loopback_policy_rejects_external_lan_and_malformed_targets(address):
    assert not is_loopback_address(address)


def test_offline_mode_sets_environment_and_blocks_external_socket(monkeypatch):
    from scripts.runtime import network_policy

    monkeypatch.delattr(socket, "_local_ai_bench_offline", raising=False)
    original_socket = socket.socket
    monkeypatch.setattr(network_policy.socket, "socket", original_socket)
    for key in OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    network_policy.apply_offline_mode()
    assert all(os.environ[key] == value for key, value in OFFLINE_ENVIRONMENT.items())
    blocked = network_policy.socket.socket()
    with pytest.raises(OfflineNetworkError):
        blocked.connect(("8.8.8.8", 53))
    blocked.close()


def test_offline_mode_blocks_udp_and_sendmsg_egress(monkeypatch):
    from scripts.runtime import network_policy

    monkeypatch.delattr(socket, "_local_ai_bench_offline", raising=False)
    monkeypatch.setattr(network_policy.socket, "socket", socket.socket)
    network_policy.apply_offline_mode()
    blocked = network_policy.socket.socket(type=socket.SOCK_DGRAM)
    with pytest.raises(OfflineNetworkError):
        blocked.sendto(b"query", ("8.8.8.8", 53))
    if hasattr(blocked, "sendmsg"):
        with pytest.raises(OfflineNetworkError):
            blocked.sendmsg([b"query"], [], 0, ("8.8.8.8", 53))
    blocked.close()

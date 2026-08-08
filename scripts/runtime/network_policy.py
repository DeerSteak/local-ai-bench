"""Loopback-only runtime policy for offline benchmark execution."""

import ipaddress
import os
import socket


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_DISABLED": "true",
}


class OfflineNetworkError(ConnectionError):
    pass


def is_loopback_address(address) -> bool:
    if isinstance(address, str):
        return True
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0].decode() if isinstance(address[0], bytes) else str(address[0])
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def apply_offline_mode() -> None:
    os.environ.update(OFFLINE_ENVIRONMENT)
    if getattr(socket, "_local_ai_bench_offline", False):
        return
    original_socket = socket.socket

    class LoopbackSocket(original_socket):
        def connect(self, address):
            if not is_loopback_address(address):
                raise OfflineNetworkError(f"offline mode blocked outbound connection: {address}")
            return super().connect(address)

        def connect_ex(self, address):
            if not is_loopback_address(address):
                raise OfflineNetworkError(f"offline mode blocked outbound connection: {address}")
            return super().connect_ex(address)

        def sendto(self, data, *args):
            address = args[-1] if args else None
            if not is_loopback_address(address):
                raise OfflineNetworkError(f"offline mode blocked outbound datagram: {address}")
            return super().sendto(data, *args)

        def sendmsg(self, buffers, ancdata=(), flags=0, address=None):
            if address is not None and not is_loopback_address(address):
                raise OfflineNetworkError(f"offline mode blocked outbound message: {address}")
            if address is None:
                return super().sendmsg(buffers, ancdata, flags)
            return super().sendmsg(buffers, ancdata, flags, address)

    socket.socket = LoopbackSocket
    setattr(socket, "_local_ai_bench_offline", True)

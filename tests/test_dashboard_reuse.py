import io
import json
from urllib.error import URLError

from scripts.app.dashboard_reuse import dashboard_server_running, reopen_dashboard


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def response(payload):
    return Response(json.dumps(payload).encode("utf-8"))


def test_dashboard_server_running_requires_workspace_config_token():
    assert dashboard_server_running(3000, open_url=lambda *_args, **_kwargs: response({
        "token": "secret",
    }))
    assert not dashboard_server_running(3000, open_url=lambda *_args, **_kwargs: response({}))
    assert not dashboard_server_running(3000, open_url=lambda *_args, **_kwargs: response([]))


def test_dashboard_server_running_treats_unavailable_port_as_not_running():
    def unavailable(*_args, **_kwargs):
        raise URLError("connection refused")

    assert not dashboard_server_running(3000, open_url=unavailable)


def test_reopen_dashboard_opens_selected_results_on_owned_server():
    opened = []
    assert reopen_dashboard(
        4321, "/?autoload=1",
        open_url=lambda *_args, **_kwargs: response({"token": "secret"}),
        open_browser=opened.append,
    )
    assert opened == ["http://127.0.0.1:4321/?autoload=1"]


def test_reopen_dashboard_does_not_open_unrelated_server():
    opened = []
    assert not reopen_dashboard(
        3000, "/", open_url=lambda *_args, **_kwargs: response({"service": "other"}),
        open_browser=opened.append,
    )
    assert opened == []

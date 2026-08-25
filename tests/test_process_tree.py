import signal
import subprocess

from scripts.runtime import process_tree


class Child:
    def __init__(self, pid):
        self.pid = pid
        self.calls = []

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


class Process:
    pid = 100

    def __init__(self, running=True):
        self.running = running
        self.calls = []

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.running:
            raise subprocess.TimeoutExpired("runner", timeout or 0)
        return 0

    def kill(self):
        self.calls.append(("kill",))
        self.running = False


def test_stop_process_tree_captures_descendants_before_signaling_parent(monkeypatch):
    child, stubborn = Child(101), Child(102)
    order = []

    class Parent:
        @staticmethod
        def children(recursive):
            assert recursive is True
            order.append("captured")
            return [child, stubborn]

    monkeypatch.setattr(process_tree.psutil, "Process", lambda pid: Parent())
    monkeypatch.setattr(
        process_tree.os, "killpg",
        lambda pid, sig: order.append(("signal", pid, sig)),
    )
    monkeypatch.setattr(
        process_tree.psutil, "wait_procs",
        lambda children, timeout: (children[:1], children[1:]),
    )
    process = Process()
    process_tree.stop_process_tree(process, timeout=2, system="Linux")
    assert order == ["captured", ("signal", 100, signal.SIGINT)]
    assert child.calls == ["terminate"]
    assert stubborn.calls == ["terminate", "kill"]
    assert process.calls == [("wait", 2), ("kill",), ("wait", None)]


def test_stop_process_tree_cleans_children_after_parent_already_exited(monkeypatch):
    child = Child(101)

    class Parent:
        @staticmethod
        def children(recursive):
            return [child]

    monkeypatch.setattr(process_tree.psutil, "Process", lambda pid: Parent())
    monkeypatch.setattr(process_tree.psutil, "wait_procs", lambda children, timeout: (children, []))
    process = Process(running=False)
    process_tree.stop_process_tree(process, timeout=2, system="Linux")
    assert child.calls == ["terminate"]
    assert process.calls == []

"""Engine Management screen composition."""

import platform
import shutil
import uuid

from scripts.runtime import config
from scripts.app.engine_management import (
    build_engine_management_tab, collect_engine_management, vllm_update_support,
)
from scripts.runtime.engines import get_engine
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.setup.model_compatibility import ModelCompatibility, probe_llamacpp_load
from scripts.setup.runtime_update import (
    RuntimeUpdateResult, detect_nvidia_max_cuda_version, fetch_llamacpp_release,
    fetch_llamacpp_release_tag, rebuild_managed_llamacpp, update_macos_llamacpp,
    update_managed_vllm, update_windows_llamacpp,
)
from scripts.setup.directory_transaction import (
    DirectorySwapError, DirectorySwapSpec, swap_staged_directories,
)


class EngineUpdateActions:
    def __init__(self, setup, hardware_backend):
        self.setup = setup
        self.hardware_backend = hardware_backend

    def update_vllm(self, control):
        return self.update_vllm_version(None, control)

    def update_vllm_version(self, version, control):
        snapshot = collect_engine_management(get_engine, self.hardware_backend)
        status = next(item for item in snapshot.statuses if item.engine == "vllm")
        support = vllm_update_support(status, self.setup, platform.machine())
        if support is None:
            return RuntimeUpdateResult(False, "This vLLM runtime is not app managed or updateable.")
        return update_managed_vllm(
            support, config.VLLM_VENV, control=control, log=control.log, version=version,
        )

    def update_llamacpp(self, control):
        return self.update_llamacpp_version(None, control)

    def update_llamacpp_version(self, tag, control):
        release_fetcher = (lambda: fetch_llamacpp_release_tag(tag)) \
            if tag else fetch_llamacpp_release
        snapshot = collect_engine_management(get_engine, self.hardware_backend)
        statuses = [
            item for item in snapshot.statuses
            if item.engine in {"llamacpp", "llamacpp-vulkan"}
            and (item.managed or (item.engine == "llamacpp" and platform.system() == "Darwin"))
        ]
        if not statuses:
            return RuntimeUpdateResult(False, "This llama.cpp runtime is not app managed.")
        release = release_fetcher()
        selected_release = lambda: release
        token = uuid.uuid4().hex
        staged_runtimes = []
        versions = []
        try:
            for status in statuses:
                control.log(f"Staging {status.engine} from the selected llama.cpp release...")
                target = (config.LLAMACPP_VULKAN_DIR if status.engine == "llamacpp-vulkan"
                          else config.LLAMACPP_DIR)
                candidate = target.with_name(f".{target.name}-family-stage-{token}")
                candidate.mkdir(parents=True)
                staged_runtimes.append((status, target, candidate))
                if platform.system() == "Darwin":
                    result = update_macos_llamacpp(
                        candidate, platform.machine(), control=control,
                        release_fetcher=selected_release,
                    )
                elif platform.system() == "Windows":
                    result = update_windows_llamacpp(
                        candidate, detect_nvidia_max_cuda_version(), control=control,
                        release_fetcher=selected_release,
                        intel_xpu=status.engine == "llamacpp" and self.hardware_backend == "xpu",
                        vulkan=status.engine == "llamacpp-vulkan",
                    )
                else:
                    result = rebuild_managed_llamacpp(
                        candidate, status.backend, control=control, log=control.log,
                        release_fetcher=selected_release,
                    )
                if not result.success:
                    return RuntimeUpdateResult(
                        False, f"{status.engine} staging failed; no runtime was replaced: "
                        f"{result.detail}",
                    )
                if result.version:
                    versions.append(result.version)
            specs = [
                DirectorySwapSpec(
                    target, candidate, target.with_name(f".{target.name}-family-backup-{token}"),
                    target.is_dir(),
                )
                for _status, target, candidate in staged_runtimes
            ]
            try:
                cleanup_errors = swap_staged_directories(
                    specs, replace=lambda source, destination: source.replace(destination),
                    remove=shutil.rmtree,
                )
            except DirectorySwapError as exc:
                return RuntimeUpdateResult(
                    False, f"llama.cpp family replacement failed; prior runtimes were restored: {exc}",
                )
            names = [status.engine for status, _target, _candidate in staged_runtimes]
            version = versions[0] if versions and len(set(versions)) == 1 else None
            detail = f"Updated all app-managed llama.cpp runtimes: {', '.join(names)}."
            if cleanup_errors:
                detail += " One or more rollback backups could not be removed."
            return RuntimeUpdateResult(True, detail, version)
        finally:
            for _status, _target, candidate in staged_runtimes:
                if candidate.exists():
                    shutil.rmtree(candidate, ignore_errors=True)

    def probe_llamacpp_model(self, tag, control):
        engine = get_engine("llamacpp")
        paths = getattr(engine, "model_paths", lambda _tag: ())(tag)
        if not paths:
            return ModelCompatibility(
                "llamacpp", tag, None, "unavailable", f"Model files for {tag} were not found.",
            )
        return probe_llamacpp_load(
            tag, paths[0], getattr(engine, "runtime_location", lambda: None)(), control=control,
        )


def build_engine_screen(notebook, *, ttk, **management_options):
    frame = ttk.Frame(notebook, padding=18)
    notebook.add(frame, text="Engine Management")
    controller = build_engine_management_tab(parent=frame, ttk=ttk, **management_options)
    return frame, controller

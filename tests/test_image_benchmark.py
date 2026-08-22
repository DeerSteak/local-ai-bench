from pathlib import Path

import pytest

from scripts.runtime import config
from scripts.workloads.image_benchmark import (
    ImageBenchmark, display_image_path, image_resume_artifacts, image_resume_runtimes,
)
from scripts.results.image_event_stage import ImageEventStage
from scripts.results.run_plan import RunPlan
from scripts.runtime.shared import Shared


def _build_kwargs(**overrides) -> dict:
    kwargs: dict = dict(checkpoint="ckpt.safetensors", width=1024, height=1024,
                        steps=20, cfg=1.0, sampler="euler", scheduler="simple",
                        seed=42, prompt="a cat", filename_prefix="prefix")
    kwargs.update(overrides)
    return kwargs


def test_build_workflow_routes_flux_type():
    wf = ImageBenchmark.build_workflow("flux", **_build_kwargs())
    assert wf == ImageBenchmark.build_flux_workflow(**_build_kwargs())


def test_build_workflow_routes_flux2_type():
    wf = ImageBenchmark.build_workflow("flux2", **_build_kwargs())
    assert wf == ImageBenchmark.build_flux2_workflow(**_build_kwargs())


def test_build_workflow_routes_z_image_type():
    wf = ImageBenchmark.build_workflow("z_image", **_build_kwargs())
    assert wf == ImageBenchmark.build_z_image_workflow(**_build_kwargs())


def test_build_workflow_falls_back_to_sdxl_for_unrecognized_type():
    for workflow_t in ("sdxl", "unknown-type", None):
        wf = ImageBenchmark.build_workflow(workflow_t, **_build_kwargs())
        assert wf == ImageBenchmark.build_sdxl_workflow(**_build_kwargs())


def test_flux_workflow_wires_checkpoint_and_prompt():
    wf = ImageBenchmark.build_flux_workflow(
        checkpoint="flux1-dev.safetensors", width=1024, height=1024,
        steps=20, cfg=1.0, sampler="euler", scheduler="simple",
        seed=42, prompt="a cat",
    )
    assert wf["1"]["inputs"]["ckpt_name"] == "flux1-dev.safetensors"
    assert wf["2"]["inputs"]["text"] == "a cat"
    assert wf["4"]["inputs"]["width"] == 1024
    assert wf["4"]["inputs"]["height"] == 1024
    assert wf["5"]["inputs"]["noise_seed"] == 42
    # Every node referenced by ["node_id", slot] must exist in the graph.
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf


def test_flux2_workflow_wires_checkpoint_and_prompt():
    wf = ImageBenchmark.build_flux2_workflow(
        checkpoint="flux2-dev.safetensors", width=1024, height=1024,
        steps=28, cfg=4.0, sampler="euler", scheduler="simple",
        seed=42, prompt="a dog",
    )
    assert wf["1"]["inputs"]["ckpt_name"] == "flux2-dev.safetensors"
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf


def test_z_image_workflow_matches_official_core_node_graph():
    wf = ImageBenchmark.build_z_image_workflow(
        checkpoint="z_image_turbo_bf16.safetensors", width=1024, height=1536,
        steps=8, cfg=1.0, sampler="res_multistep", scheduler="simple",
        seed=42, prompt="a cat",
    )
    assert wf["1"] == {"class_type": "UNETLoader", "inputs": {
        "unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default",
    }}
    assert wf["2"]["inputs"] == {
        "clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default",
    }
    assert wf["3"]["inputs"]["vae_name"] == "ae.safetensors"
    assert wf["4"]["inputs"]["text"] == "a cat"
    assert wf["5"] == {"class_type": "ConditioningZeroOut", "inputs": {
        "conditioning": ["4", 0],
    }}
    assert wf["6"]["inputs"] == {"width": 1024, "height": 1536, "batch_size": 1}
    assert wf["7"]["inputs"] == {"model": ["1", 0], "shift": 3.0}
    assert wf["8"]["inputs"] == {
        "model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0],
        "latent_image": ["6", 0], "seed": 42, "steps": 8, "cfg": 1.0,
        "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
    }
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf


def test_sdxl_workflow_wires_checkpoint_and_prompt():
    wf = ImageBenchmark.build_sdxl_workflow(
        checkpoint="sd_xl_base_1.0.safetensors", width=1024, height=1024,
        steps=20, cfg=7.0, sampler="euler_ancestral", scheduler="normal",
        seed=42, prompt="a cat",
    )
    assert wf["4"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    assert wf["6"]["inputs"]["text"] == "a cat"
    assert wf["7"]["inputs"]["text"] == ""  # negative prompt is empty
    for node in wf.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf


def test_flux_and_flux2_use_different_filename_prefixes():
    wf1 = ImageBenchmark.build_flux_workflow(
        checkpoint="c", width=8, height=8, steps=1, cfg=1.0,
        sampler="euler", scheduler="simple", seed=1, prompt="p",
    )
    wf2 = ImageBenchmark.build_flux2_workflow(
        checkpoint="c", width=8, height=8, steps=1, cfg=1.0,
        sampler="euler", scheduler="simple", seed=1, prompt="p",
    )
    save1 = [n for n in wf1.values() if n["class_type"] == "SaveImage"][0]
    save2 = [n for n in wf2.values() if n["class_type"] == "SaveImage"][0]
    assert save1["inputs"]["filename_prefix"] != save2["inputs"]["filename_prefix"]


def test_image_resume_inputs_include_existing_workflow_assets_only(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    for folder, name in (
        ("checkpoints", "flux.safetensors"), ("clip", "t5xxl_fp16.safetensors"),
        ("clip", "clip_l.safetensors"), ("vae", "ae.safetensors"),
    ):
        path = tmp_path / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    model = {
        "short": "flux", "checkpoint": "flux.safetensors", "workflow": "flux",
        "support_assets": [
            {"folder": "clip", "name": "t5xxl_fp16.safetensors"},
            {"folder": "clip", "name": "clip_l.safetensors"},
            {"folder": "vae", "name": "ae.safetensors"},
        ],
    }
    artifacts = image_resume_artifacts([model])
    assert set(artifacts) == {
        "image:flux:checkpoint", "image:flux:clip:t5xxl_fp16.safetensors",
        "image:flux:clip:clip_l.safetensors", "image:flux:vae:ae.safetensors",
    }


def test_z_image_resume_inputs_use_diffusion_model_and_declared_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    model = {
        "short": "z-image", "checkpoint": "z.safetensors",
        "checkpoint_folder": "diffusion_models", "workflow": "z_image",
        "support_assets": [
            {"folder": "text_encoders", "name": "qwen.safetensors"},
            {"folder": "vae", "name": "ae.safetensors"},
        ],
    }
    for folder, name in (("diffusion_models", "z.safetensors"),
                         ("text_encoders", "qwen.safetensors"),
                         ("vae", "ae.safetensors")):
        path = tmp_path / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())

    assert set(image_resume_artifacts([model])) == {
        "image:z-image:checkpoint",
        "image:z-image:text_encoders:qwen.safetensors",
        "image:z-image:vae:ae.safetensors",
    }


def test_image_resume_runtime_uses_selected_comfyui_install(monkeypatch, tmp_path):
    main = tmp_path / "main.py"
    python = tmp_path / "python"
    main.write_bytes(b"main")
    python.write_bytes(b"python")
    monkeypatch.setattr(Shared, "find_comfyui_python", lambda _path: str(python))
    assert image_resume_runtimes(tmp_path) == {
        "comfyui-main": main, "comfyui-python": python,
    }


def test_image_display_path_tolerates_output_outside_repository(tmp_path):
    assert display_image_path(tmp_path / "image.png") == tmp_path / "image.png"


def test_representative_image_falls_back_to_comfyui_output(monkeypatch, tmp_path):
    source = tmp_path / "output" / "nested" / "generated.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fallback-png")
    destination = tmp_path / "images" / "saved.png"
    monkeypatch.setattr(
        ImageBenchmark, "save_comfyui_image",
        staticmethod(lambda *_args: (_ for _ in ()).throw(ConnectionError("offline"))),
    )
    assert ImageBenchmark.save_representative_image(
        {"filename": "generated.png", "subfolder": "nested"}, destination, tmp_path,
    ) is True
    assert destination.read_bytes() == b"fallback-png"


def test_comfyui_free_models_posts_unload_and_free_memory(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        class _Resp:
            ok = True
        return _Resp()

    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.post", fake_post)
    ImageBenchmark.comfyui_free_models(timeout=5)

    assert len(calls) == 1
    url, payload, timeout = calls[0]
    assert url == f"{config.COMFYUI_URL}/free"
    assert payload == {"unload_models": True, "free_memory": True}
    assert timeout == 5


def test_comfyui_free_models_swallows_request_errors(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise ConnectionError("comfyui unreachable")

    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.post", fake_post)
    # Should not raise even though the request fails.
    ImageBenchmark.comfyui_free_models()


def test_run_attaches_model_memory_with_resolution_subwindows(monkeypatch, tmp_path):
    events = []

    class Telemetry:
        def __init__(self):
            self.calls = []
            self.last_power = {"energy_joules": 2}
        def begin_model_load(self): self.calls.append("load")
        def begin_measured(self, name): self.calls.append(name)
        def begin_pause(self): self.calls.append("pause"); events.append("pause-window")
        def finish_case(self):
            self.calls.append("finish")
            return {"windows": [{"name": name} for name in self.calls if name.startswith("measured:")]}

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "N_RUNS", 1)
    monkeypatch.setattr(ImageBenchmark, "comfyui_submit", staticmethod(lambda *_a, **_k: (1.0, [])))
    monkeypatch.setattr(ImageBenchmark, "comfyui_free_models", staticmethod(lambda: None))
    monkeypatch.setattr(
        "scripts.workloads.image_benchmark.wait_if_paused", lambda: events.append("wait"),
    )
    telemetry = Telemetry()
    model = {
        "label": "Image", "checkpoint": "model.safetensors", "workflow": "sdxl",
        "steps": 1, "cfg": 1.0, "sampler": "euler", "scheduler": "normal",
        "short": "image", "resolutions": [(64, 64), (128, 128)],
    }
    result = ImageBenchmark().run(
        [model], [(64, 64)], 1, "prompt", tmp_path,
        images_dir=tmp_path / "images", telemetry=telemetry,
    )
    assert telemetry.calls == [
        "load", "pause", "measured:64x64", "pause", "measured:128x128", "finish",
    ]
    assert events == ["pause-window", "wait", "pause-window", "wait"]
    assert [window["name"] for window in result["image"]["memory"]["windows"]] == [
        "measured:64x64", "measured:128x128",
    ]
    assert result["image"]["power"]["efficiency"] == {
        "unit": "images_per_joule", "work_count": 2, "per_joule": 1,
    }


def test_comfyui_interrupt_and_clear_stops_once_queue_is_empty(monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append(url)
        class _Resp:
            pass
        return _Resp()

    def fake_get(url, timeout=None):
        class _Resp:
            @staticmethod
            def json():
                return {"queue_running": [], "queue_pending": []}
        return _Resp()

    sleeps = []
    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.post", fake_post)
    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.get", fake_get)
    monkeypatch.setattr("scripts.workloads.image_benchmark.time.sleep", lambda s: sleeps.append(s))

    ImageBenchmark.comfyui_interrupt_and_clear(timeout=5, confirm_timeout=15)

    assert f"{config.COMFYUI_URL}/interrupt" in posts
    assert f"{config.COMFYUI_URL}/queue" in posts
    # Queue was already empty on the first poll, so no need to sleep and retry.
    assert sleeps == []


def test_comfyui_interrupt_and_clear_polls_until_queue_drains(monkeypatch):
    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.post", lambda *a, **k: None)

    responses = [
        {"queue_running": [{"id": 1}], "queue_pending": []},
        {"queue_running": [], "queue_pending": []},
    ]

    def fake_get(url, timeout=None):
        class _Resp:
            @staticmethod
            def json():
                return responses.pop(0)
        return _Resp()

    sleeps = []
    monkeypatch.setattr("scripts.workloads.image_benchmark.requests.get", fake_get)
    monkeypatch.setattr("scripts.workloads.image_benchmark.time.sleep", lambda s: sleeps.append(s))

    ImageBenchmark.comfyui_interrupt_and_clear(timeout=5, confirm_timeout=15)

    assert sleeps == [1]
    assert responses == []


def test_handle_crashed_warmup_returns_true_when_restart_succeeds(monkeypatch):
    monkeypatch.setattr(Shared, "ensure_comfyui", lambda comfyui_dir: True)
    monkeypatch.setattr(Shared, "tail_comfyui_log", lambda: "some log output")

    assert ImageBenchmark.handle_crashed_warmup(Path("/fake/ComfyUI"), "Z-Image Turbo") is True


def test_handle_crashed_warmup_returns_false_when_restart_fails(monkeypatch):
    monkeypatch.setattr(Shared, "ensure_comfyui", lambda comfyui_dir: False)
    monkeypatch.setattr(Shared, "tail_comfyui_log", lambda: "some log output")

    assert ImageBenchmark.handle_crashed_warmup(Path("/fake/ComfyUI"), "Z-Image Turbo") is False


def test_handle_crashed_warmup_passes_comfyui_dir_through_to_restart(monkeypatch):
    seen = []
    monkeypatch.setattr(Shared, "ensure_comfyui", lambda comfyui_dir: seen.append(comfyui_dir) or True)
    monkeypatch.setattr(Shared, "tail_comfyui_log", lambda: "")

    ImageBenchmark.handle_crashed_warmup(Path("/some/ComfyUI"), "Flux.1-dev")

    assert seen == [Path("/some/ComfyUI")]


@pytest.mark.parametrize("interrupt_index", [0, 1, 2])
def test_journal_resume_reruns_only_unfinished_image_resolutions(
        monkeypatch, tmp_path, interrupt_index):
    resolutions = [(64, 64), (96, 96), (128, 128)]
    model = {
        "label": "Image", "checkpoint": "model.safetensors", "workflow": "sdxl",
        "steps": 1, "cfg": 1.0, "sampler": "euler", "scheduler": "normal",
        "short": "image", "resolutions": resolutions,
    }
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "image"}],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "N_RUNS", 1)
    monkeypatch.setattr(ImageBenchmark, "comfyui_free_models", staticmethod(lambda: None))
    monkeypatch.setattr("scripts.workloads.image_benchmark.wait_if_paused", lambda: None)
    path = tmp_path / "events.sqlite3"
    first = ImageEventStage(path, plan, lambda _: None, resume_identity=identity)
    measured = []

    def interrupted_submit(workflow, **_kwargs):
        prefix = next(node["inputs"]["filename_prefix"] for node in workflow.values()
                      if node["class_type"] == "SaveImage")
        if "warmup" not in prefix:
            measured.append(prefix)
            if prefix.startswith(f"image_{resolutions[interrupt_index][0]}x"):
                raise KeyboardInterrupt
        return 1.0, []

    monkeypatch.setattr(ImageBenchmark, "comfyui_submit", staticmethod(interrupted_submit))
    with pytest.raises(KeyboardInterrupt):
        ImageBenchmark().run(
            [model], resolutions, 1, "prompt", tmp_path,
            images_dir=tmp_path / "images", journal=first,
        )
    first.close()
    resumed = ImageEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
    )
    resumed_calls = []

    def resumed_submit(workflow, **_kwargs):
        prefix = next(node["inputs"]["filename_prefix"] for node in workflow.values()
                      if node["class_type"] == "SaveImage")
        if "warmup" not in prefix:
            resumed_calls.append(prefix)
        return 1.0, []

    monkeypatch.setattr(ImageBenchmark, "comfyui_submit", staticmethod(resumed_submit))
    result = ImageBenchmark().run(
        [model], resolutions, 1, "prompt", tmp_path,
        images_dir=tmp_path / "images", journal=resumed,
    )
    resumed.close()
    assert resumed_calls == [
        f"image_{width}x{height}_run1" for width, height in resolutions[interrupt_index:]
    ]
    assert list(result["image"]["resolutions"]) == [
        f"{width}x{height}" for width, height in resolutions
    ]


def test_journal_commits_content_addressed_png_after_visible_save(monkeypatch, tmp_path):
    model = {
        "label": "Image", "checkpoint": "model.safetensors", "workflow": "sdxl",
        "steps": 1, "cfg": 1.0, "sampler": "euler", "scheduler": "normal",
        "short": "image", "resolutions": [(64, 64)],
    }
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "image"}],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "N_RUNS", 1)
    monkeypatch.setattr(
        ImageBenchmark, "comfyui_submit",
        staticmethod(lambda *_args, **_kwargs: (
            1.0, [{"filename": "generated.png", "type": "output"}],
        )),
    )
    monkeypatch.setattr(
        ImageBenchmark, "save_comfyui_image",
        staticmethod(lambda _image, dest: (dest.parent.mkdir(parents=True, exist_ok=True),
                                           dest.write_bytes(b"png-bytes"))),
    )
    monkeypatch.setattr(ImageBenchmark, "comfyui_free_models", staticmethod(lambda: None))
    monkeypatch.setattr("scripts.workloads.image_benchmark.wait_if_paused", lambda: None)
    images_dir = tmp_path / "images"
    stage = ImageEventStage(tmp_path / "events.sqlite3", plan, lambda _: None)
    ImageBenchmark().run(
        [model], [(64, 64)], 1, "prompt", tmp_path,
        images_dir=images_dir, journal=stage,
    )
    case = next(case for case in stage.store.rebuild(plan.job_id)["cases"].values()
                if case.get("case_kind") == "image_resolution")
    assert case["artifact"] == {
        "sha256": "ea80334363eed145dfeee51ebae7dc3f1cd7d0c7879f8bfd2070c061d3c33f56",
        "size": 9, "media_type": "image/png",
    }
    digest = case["artifact"]["sha256"]
    assert (images_dir / ".artifacts" / digest[:2] / digest[2:]).is_file()
    stage.close()

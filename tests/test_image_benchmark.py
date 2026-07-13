import config
from image_benchmark import ImageBenchmark


def _build_kwargs(**overrides):
    kwargs = dict(checkpoint="ckpt.safetensors", width=1024, height=1024,
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


def test_build_workflow_routes_sd3_type():
    wf = ImageBenchmark.build_workflow("sd3", **_build_kwargs())
    assert wf == ImageBenchmark.build_sd3_workflow(**_build_kwargs())


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


def test_sd3_workflow_wires_checkpoint_and_prompt():
    wf = ImageBenchmark.build_sd3_workflow(
        checkpoint="sd3.5_large.safetensors", width=1024, height=1024,
        steps=28, cfg=4.5, sampler="euler", scheduler="beta",
        seed=42, prompt="a cat",
    )
    assert wf["1"]["inputs"]["ckpt_name"] == "sd3.5_large.safetensors"
    assert wf["3"]["inputs"]["text"] == "a cat"
    assert wf["4"]["inputs"]["text"] == ""  # negative prompt is empty
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


def test_comfyui_free_models_posts_unload_and_free_memory(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        class _Resp:
            ok = True
        return _Resp()

    monkeypatch.setattr("image_benchmark.requests.post", fake_post)
    ImageBenchmark.comfyui_free_models(timeout=5)

    assert len(calls) == 1
    url, payload, timeout = calls[0]
    assert url == f"{config.COMFYUI_URL}/free"
    assert payload == {"unload_models": True, "free_memory": True}
    assert timeout == 5


def test_comfyui_free_models_swallows_request_errors(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise ConnectionError("comfyui unreachable")

    monkeypatch.setattr("image_benchmark.requests.post", fake_post)
    # Should not raise even though the request fails.
    ImageBenchmark.comfyui_free_models()


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
    monkeypatch.setattr("image_benchmark.requests.post", fake_post)
    monkeypatch.setattr("image_benchmark.requests.get", fake_get)
    monkeypatch.setattr("image_benchmark.time.sleep", lambda s: sleeps.append(s))

    ImageBenchmark.comfyui_interrupt_and_clear(timeout=5, confirm_timeout=15)

    assert f"{config.COMFYUI_URL}/interrupt" in posts
    assert f"{config.COMFYUI_URL}/queue" in posts
    # Queue was already empty on the first poll, so no need to sleep and retry.
    assert sleeps == []


def test_comfyui_interrupt_and_clear_polls_until_queue_drains(monkeypatch):
    monkeypatch.setattr("image_benchmark.requests.post", lambda *a, **k: None)

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
    monkeypatch.setattr("image_benchmark.requests.get", fake_get)
    monkeypatch.setattr("image_benchmark.time.sleep", lambda s: sleeps.append(s))

    ImageBenchmark.comfyui_interrupt_and_clear(timeout=5, confirm_timeout=15)

    assert sleeps == [1]
    assert responses == []

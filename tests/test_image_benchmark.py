from image_benchmark import ImageBenchmark


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

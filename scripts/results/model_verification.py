"""Verify resumed model weights at the first pending load in each stage."""

from scripts.results.resume_policy import file_identity


def verify_resume_model(journal, model: dict, engine=None, *, progress=None) -> None:
    if journal is None:
        return
    if not any(event.payload.get("recovery") in {"resume", "retry"}
               for event in journal.store.events(journal.plan.job_id)):
        return
    prefix = f"model:{model['tag']}:" if engine is not None else f"image:{model['short']}:"
    verified = getattr(journal, "_verified_model_weights", set())
    if prefix in verified:
        return
    saved = journal.store.resume_identity(journal.plan.job_id).get("artifacts", {})
    expected = {name: identity for name, identity in saved.items() if name.startswith(prefix)}
    if engine is None:
        from scripts.workloads.image_benchmark import image_resume_artifacts
        paths = image_resume_artifacts([model])
    else:
        paths = {f"{prefix}part{index}": path for index, path in enumerate(
            engine.resume_artifact_paths(model["tag"]) if engine.model_pulled(model["tag"]) else (), 1)}
    if paths.keys() != expected.keys():
        raise ValueError(f"Model artifacts changed for {model['short']}; create a fork")
    for name, path in paths.items():
        if file_identity(path, progress=progress) != expected[name]:
            raise ValueError(f"Model bytes changed for {model['short']}; create a fork")
    verified.add(prefix)
    journal._verified_model_weights = verified

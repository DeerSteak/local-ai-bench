"""image_benchmark.py — ComfyUI-driven image generation benchmark."""

import time
import shutil
from pathlib import Path

import requests

from scripts.runtime import config
from scripts.results.content_store import ContentStore
from scripts.runtime.shared import Shared
from scripts.runtime.progress_events import emit_model_finished, emit_progress
from scripts.runtime.pause_control import wait_if_paused
from scripts.runtime.telemetry import add_power_efficiency
from scripts.workloads.models import image_checkpoint_folder, image_checkpoint_path


def image_resume_artifacts(models: list[dict]) -> dict[str, Path]:
    """Return existing selected image inputs under path-free logical names."""
    artifacts = {}
    for model in models:
        short = model["short"]
        paths = [("checkpoint", image_checkpoint_folder(model), model["checkpoint"])]
        paths.extend((f"{asset['folder']}:{asset['name']}", asset["folder"], asset["name"])
                     for asset in model.get("support_assets", ()))
        for logical, folder, name in paths:
            path = config.COMFYUI_MODELS_DIR / folder / name
            if path.is_file():
                artifacts[f"image:{short}:{logical}"] = path
    return artifacts


def image_resume_runtimes(comfyui_dir: Path) -> dict[str, Path]:
    """Return existing ComfyUI entrypoint/interpreter files for resume identity."""
    runtimes = {}
    main = Path(comfyui_dir) / "main.py"
    if main.is_file():
        runtimes["comfyui-main"] = main
    try:
        python = Path(Shared.find_comfyui_python(Path(comfyui_dir)))
    except (OSError, TypeError, ValueError):
        python = None
    if python is not None and python.is_file():
        runtimes["comfyui-python"] = python
    return runtimes


def display_image_path(path: Path) -> Path:
    try:
        return path.relative_to(config.SCRIPT_DIR)
    except ValueError:
        return path


class ImageBenchmark:
    @staticmethod
    def comfyui_free_models(timeout: int = 10) -> None:
        """Unload whatever checkpoint(s) ComfyUI has resident — see docs/workloads.md's image-generation section."""
        try:
            requests.post(f"{config.COMFYUI_URL}/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=timeout)
        except Exception as e:
            Shared.warn(f"Could not unload ComfyUI models: {e}")

    @staticmethod
    def comfyui_interrupt_and_clear(timeout: int = 10, confirm_timeout: int = 15) -> None:
        """Stop ComfyUI's running job and drop the queue, then poll until both
        are empty — see docs/workloads.md's image-generation section."""
        try:
            requests.post(f"{config.COMFYUI_URL}/interrupt", timeout=timeout)
        except Exception as e:
            Shared.warn(f"Failed to interrupt ComfyUI job: {e}")
        try:
            requests.post(f"{config.COMFYUI_URL}/queue", json={"clear": True}, timeout=timeout)
        except Exception as e:
            Shared.warn(f"Failed to clear ComfyUI queue: {e}")

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < confirm_timeout:
            try:
                status = requests.get(f"{config.COMFYUI_URL}/queue", timeout=10).json()
            except Exception as e:
                Shared.warn(f"Failed to confirm ComfyUI queue is clear: {e}")
                return
            if not status.get("queue_running") and not status.get("queue_pending"):
                return
            time.sleep(1)
        Shared.warn(f"ComfyUI queue still not empty {confirm_timeout}s after interrupt/clear — "
                    f"a stuck job may still be occupying the execution slot")

    @staticmethod
    def build_flux_workflow(checkpoint, width, height, steps, cfg,
                            sampler, scheduler, seed, prompt, filename_prefix="bench_flux"):
        """Flux.1 txt2img workflow. BFL's flux1 .safetensors are transformer-only
        (no CLIP/VAE), so those load via separate nodes, not checkpoint output slots 1/2."""
        return {
            # UNet from checkpoint (output 0 = model; slots 1/2 are None for BFL files)
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": checkpoint}},
            # Dual CLIP for Flux: T5-XXL + CLIP-L
            "12": {"class_type": "DualCLIPLoader",
                   "inputs": {
                       "clip_name1": "t5xxl_fp16.safetensors",
                       "clip_name2": "clip_l.safetensors",
                       "type": "flux",
                   }},
            # VAE loaded separately
            "13": {"class_type": "VAELoader",
                   "inputs": {"vae_name": "ae.safetensors"}},
            # Encode prompt using dual CLIP — no negative for Flux
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["12", 0]}},
            # Flux guidance node (replaces CFGGuider)
            "3": {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["2", 0], "guidance": cfg}},
            # Empty latent image
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            # Noise source
            "5": {"class_type": "RandomNoise",
                  "inputs": {"noise_seed": seed}},
            # Basic guider wrapping FluxGuidance conditioning
            "6": {"class_type": "BasicGuider",
                  "inputs": {"model": ["1", 0], "conditioning": ["3", 0]}},
            # Sampler selection
            "7": {"class_type": "KSamplerSelect",
                  "inputs": {"sampler_name": sampler}},
            # Scheduler
            "8": {"class_type": "BasicScheduler",
                  "inputs": {
                      "model": ["1", 0],
                      "scheduler": scheduler,
                      "steps": steps,
                      "denoise": 1.0,
                  }},
            # Run the sampler
            "9": {"class_type": "SamplerCustomAdvanced",
                  "inputs": {
                      "noise": ["5", 0],
                      "guider": ["6", 0],
                      "sampler": ["7", 0],
                      "sigmas": ["8", 0],
                      "latent_image": ["4", 0],
                  }},
            # Decode latent to image using separate VAE
            "10": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
            # Save
            "11": {"class_type": "SaveImage",
                   "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix}},
        }

    @staticmethod
    def build_flux2_workflow(checkpoint, width, height, steps, cfg,
                             sampler, scheduler, seed, prompt, filename_prefix="bench_flux2"):
        """Flux.2-dev txt2img workflow — its own CLIPLoader/VAE; reusing
        Flux.1's DualCLIPLoader/VAE fails silently with a dimension mismatch."""
        return {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": checkpoint}},
            "12": {"class_type": "CLIPLoader",
                   "inputs": {
                       "clip_name": "mistral_3_small_flux2_fp8.safetensors",
                       "type": "flux2",
                   }},
            "13": {"class_type": "VAELoader",
                   "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["12", 0]}},
            "3": {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["2", 0], "guidance": cfg}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "RandomNoise",
                  "inputs": {"noise_seed": seed}},
            "6": {"class_type": "BasicGuider",
                  "inputs": {"model": ["1", 0], "conditioning": ["3", 0]}},
            "7": {"class_type": "KSamplerSelect",
                  "inputs": {"sampler_name": sampler}},
            "8": {"class_type": "BasicScheduler",
                  "inputs": {
                      "model": ["1", 0],
                      "scheduler": scheduler,
                      "steps": steps,
                      "denoise": 1.0,
                  }},
            "9": {"class_type": "SamplerCustomAdvanced",
                  "inputs": {
                      "noise": ["5", 0],
                      "guider": ["6", 0],
                      "sampler": ["7", 0],
                      "sigmas": ["8", 0],
                      "latent_image": ["4", 0],
                  }},
            "10": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
            "11": {"class_type": "SaveImage",
                   "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix}},
        }

    @staticmethod
    def build_z_image_workflow(checkpoint, width, height, steps, cfg,
                               sampler, scheduler, seed, prompt,
                               filename_prefix="bench_z_image"):
        """Z-Image Turbo graph matching Comfy-Org's core-node workflow."""
        return {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader",
                  "inputs": {
                      "clip_name": "qwen_3_4b.safetensors",
                      "type": "lumina2",
                      "device": "default",
                  }},
            "3": {"class_type": "VAELoader",
                  "inputs": {"vae_name": "ae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "ConditioningZeroOut",
                  "inputs": {"conditioning": ["4", 0]}},
            "6": {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "7": {"class_type": "ModelSamplingAuraFlow",
                  "inputs": {"model": ["1", 0], "shift": 3.0}},
            "8": {"class_type": "KSampler",
                  "inputs": {
                      "model": ["7", 0], "positive": ["4", 0],
                      "negative": ["5", 0], "latent_image": ["6", 0],
                      "seed": seed, "steps": steps, "cfg": cfg,
                      "sampler_name": sampler, "scheduler": scheduler,
                      "denoise": 1.0,
                  }},
            "9": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage",
                   "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix}},
        }

    @staticmethod
    def build_sdxl_workflow(checkpoint, width, height, steps, cfg,
                            sampler, scheduler, seed, prompt, filename_prefix="bench"):
        """Minimal SDXL txt2img workflow for ComfyUI API."""
        return {
            "4":  {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": checkpoint}},
            "6":  {"class_type": "CLIPTextEncode",
                   "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7":  {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "", "clip": ["4", 1]}},
            "8":  {"class_type": "VAEDecode",
                   "inputs": {"samples": ["10", 0], "vae": ["4", 2]}},
            "9":  {"class_type": "SaveImage",
                   "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}},
            "5":  {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
            "10": {"class_type": "KSampler",
                   "inputs": {
                       "model": ["4", 0], "positive": ["6", 0],
                       "negative": ["7", 0], "latent_image": ["5", 0],
                       "seed": seed, "steps": steps, "cfg": cfg,
                       "sampler_name": sampler, "scheduler": scheduler,
                       "denoise": 1.0,
                   }},
        }

    @staticmethod
    def handle_crashed_warmup(comfyui_dir: Path, label: str) -> bool:
        """After a warmup crashes ComfyUI, try to restart it. Returns whether
        the run should proceed to the next model (False means abort entirely,
        since a dead server that won't restart will fail every remaining model too)."""
        Shared.warn(f"ComfyUI appears to have crashed — last output:\n{Shared.tail_comfyui_log()}")
        if Shared.ensure_comfyui(comfyui_dir):
            return True
        Shared.err(f"Could not restart ComfyUI after it crashed during {label}'s warmup "
                   f"— skipping remaining image models")
        return False

    @staticmethod
    def build_workflow(workflow_t, checkpoint, width, height, steps, cfg,
                       sampler, scheduler, seed, prompt, filename_prefix):
        """Route to the right workflow builder for `workflow_t`; unrecognized
        types fall through to the SDXL graph, which also works for SD1.5."""
        if workflow_t == "flux":
            builder = ImageBenchmark.build_flux_workflow
        elif workflow_t == "flux2":
            builder = ImageBenchmark.build_flux2_workflow
        elif workflow_t == "z_image":
            builder = ImageBenchmark.build_z_image_workflow
        else:
            builder = ImageBenchmark.build_sdxl_workflow
        return builder(checkpoint, width, height, steps, cfg, sampler, scheduler,
                       seed, prompt, filename_prefix=filename_prefix)

    @staticmethod
    def comfyui_submit(workflow: dict, timeout: int = 300) -> tuple[float, list[dict]]:  # pragma: no cover — submits to and polls a real ComfyUI server
        """Submit a workflow to ComfyUI, poll until done. Returns
        (elapsed_sec, images), images being {"filename", "subfolder", "type"} dicts."""
        # A stuck prior job can still be queued if its own timeout handling failed to clear it.
        try:
            queue_status = requests.get(f"{config.COMFYUI_URL}/queue", timeout=10).json()
            if queue_status.get("queue_running") or queue_status.get("queue_pending"):
                Shared.warn("ComfyUI queue has leftover job(s) from a prior submission — clearing before continuing")
                ImageBenchmark.comfyui_interrupt_and_clear()
        except Exception as e:
            Shared.warn(f"Failed to check ComfyUI queue before submission: {e}")

        resp = requests.post(
            f"{config.COMFYUI_URL}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(f"ComfyUI rejected workflow (HTTP {resp.status_code}): {detail}")
        prompt_id = resp.json()["prompt_id"]

        # Start timing AFTER submission so we measure generation time only,
        # and stale history entries from previous runs won't match this prompt_id.
        t0 = time.perf_counter()
        seen = False  # True once we see this prompt_id appear in history

        while True:
            time.sleep(1)
            try:
                status = requests.get(
                    f"{config.COMFYUI_URL}/history/{prompt_id}", timeout=10
                ).json()
            except Exception:
                if time.perf_counter() - t0 > timeout:
                    ImageBenchmark.comfyui_interrupt_and_clear()
                    raise TimeoutError(f"ComfyUI job timed out after {timeout}s")
                continue

            if prompt_id in status:
                seen = True
                job = status[prompt_id]
                job_status = job.get("status", {})

                if job_status.get("status_str") == "error" or job.get("error"):
                    msgs = job.get("error") or job_status.get("messages", [])
                    raise RuntimeError(f"ComfyUI job failed: {msgs}")

                if job_status.get("completed"):
                    elapsed = time.perf_counter() - t0
                    images = []
                    for node_out in job.get("outputs", {}).values():
                        images.extend(node_out.get("images", []))
                    return elapsed, images

            if time.perf_counter() - t0 > timeout:
                ImageBenchmark.comfyui_interrupt_and_clear()
                if not seen:
                    raise TimeoutError(
                        f"ComfyUI job never appeared in history after {timeout}s "
                        f"— may be queued behind a still-running prior job, or the "
                        f"workflow errored before queuing"
                    )
                raise TimeoutError(f"ComfyUI job timed out after {timeout}s")

    @staticmethod
    def save_comfyui_image(img: dict, dest: Path) -> None:  # pragma: no cover — fetches from a real ComfyUI server
        """Fetch a generated image from ComfyUI and save it locally."""
        resp = requests.get(
            f"{config.COMFYUI_URL}/view",
            params={
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type":     img.get("type", "output"),
            },
            timeout=30,
        )
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)

    @staticmethod
    def save_representative_image(img: dict, dest: Path, comfyui_dir: Path) -> bool:
        try:
            ImageBenchmark.save_comfyui_image(img, dest)
            Shared.ok(f"Saved image → {display_image_path(dest)}")
            return True
        except Exception as exc:
            Shared.warn(f"HTTP image fetch failed ({exc}) — trying direct file copy")
        subfolder = img.get("subfolder", "")
        source = (comfyui_dir / "output" / subfolder / img["filename"]
                  if subfolder else comfyui_dir / "output" / img["filename"])
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            Shared.ok(f"Saved image (file copy) → {display_image_path(dest)}")
            return True
        except Exception as exc:
            Shared.warn(f"Could not save image: {exc}")
            return False

    def run(self, image_models, resolutions, seed, prompt,  # pragma: no cover — orchestrates real ComfyUI runs
            comfyui_dir, timeout=None, save_fn=None, images_dir=None, telemetry=None,
            journal=None):
        if timeout is None:
            timeout = config.RUN_TIMEOUT
        if images_dir is None:
            images_dir = config.RESULTS_DIR / "images"
        results = journal.export() if journal else {}
        Shared.section("Image Generation via ComfyUI")

        for model in image_models:
            label      = model["label"]
            checkpoint = model["checkpoint"]
            workflow_t = model["workflow"]
            steps      = model["steps"]
            cfg        = model["cfg"]
            sampler    = model["sampler"]
            scheduler  = model["scheduler"]
            short      = model["short"]
            model_resolutions = model.get("resolutions", resolutions)
            pending = (journal.pending_resolutions(model, model_resolutions)
                       if journal else [(w, h, 1) for w, h in model_resolutions])
            if not pending:
                continue

            emit_progress("model", "img", "running", label, model_id=short)
            telemetry_active = False
            segment_work = 0
            try:
                ckpt_path = image_checkpoint_path(model, config.COMFYUI_MODELS_DIR)
                if not ckpt_path.exists():
                    Shared.warn(f"{label}: checkpoint not found at {ckpt_path} — skipping")
                    Shared.log(f"Download and place at: {ckpt_path}")
                    if journal:
                        journal.record_model_state(model, "skipped", {
                            "skipped": True, "skip_reason": "checkpoint_not_found",
                        })
                    continue

                Shared.ok(f"{label}: checkpoint found ({ckpt_path.stat().st_size / (1024**3):.1f} GB)")
                if not journal:
                    results[short] = {"label": label, "checkpoint": checkpoint,
                                      "steps": steps, "resolutions": {}}

                if telemetry:
                    telemetry.begin_model_load()
                    telemetry_active = True
                # Warmup: one generation at the smallest resolution to trigger Metal/CUDA
                # shader compilation before timing starts.
                w0, h0 = model_resolutions[0]
                Shared.log(f"{label}: warmup run ({w0}x{h0}, timeout: {timeout}s) ...")
                warmup_ok = True
                warmup_seed = seed - 1  # outside the measured runs' range — see docs/workloads.md
                try:
                    wf = ImageBenchmark.build_workflow(workflow_t, checkpoint, w0, h0, steps, cfg,
                                                       sampler, scheduler, warmup_seed, prompt,
                                                       filename_prefix=f"{short}_warmup")
                    ImageBenchmark.comfyui_submit(wf, timeout=timeout)
                    Shared.ok(f"{label}: warmup done")
                except Exception as e:
                    Shared.warn(f"{label}: warmup failed ({e}) — skipping")
                    warmup_ok = False
                    if not Shared.comfyui_available():
                        if not ImageBenchmark.handle_crashed_warmup(comfyui_dir, label):
                            if journal:
                                raise RuntimeError("ComfyUI could not recover after image warmup")
                            return results

                if not warmup_ok:
                    if journal:
                        for w, h, attempt_number in pending:
                            journal.record_resolution(
                                model, w, h, [], config.N_RUNS, "failed",
                                attempt_number=attempt_number, failure_detail="warmup failed",
                            )
                    continue

                img_dir = images_dir

                model_timed_out = False
                for w, h, attempt_number in pending:
                    res_label = f"{w}x{h}"
                    Shared.log(f"{label} @ {res_label} — {config.N_RUNS} runs ...")
                    times = []
                    last_images: list[dict] = []
                    resolution_status = "ok"

                    for run_i in range(config.N_RUNS):
                        if telemetry:
                            telemetry.begin_pause()
                        wait_if_paused()
                        if telemetry:
                            telemetry.begin_measured(f"measured:{res_label}")
                        try:
                            prefix = f"{short}_{res_label}_run{run_i + 1}"
                            run_seed = seed + run_i  # varied per run — see docs/workloads.md
                            wf = ImageBenchmark.build_workflow(workflow_t, checkpoint, w, h, steps, cfg,
                                                               sampler, scheduler, run_seed, prompt,
                                                               filename_prefix=prefix)

                            elapsed, images = ImageBenchmark.comfyui_submit(wf, timeout=timeout)
                            times.append(elapsed)
                            last_images = images
                            Shared.output(f"    run {run_i+1}/{config.N_RUNS}: {elapsed:.1f}s")
                        except TimeoutError:
                            Shared.err(f"Run {run_i+1} timed out — skipping {label}")
                            model_timed_out = True
                            resolution_status = "timed_out"
                            if not journal:
                                results[short]["timed_out"] = res_label
                            break
                        except Exception as e:
                            Shared.err(f"Run {run_i+1} failed: {e}")

                    if times and not journal:
                        results[short]["resolutions"][res_label] = {
                            "sec_per_image_mean":  round(Shared.mean(times),  2),
                            "sec_per_image_stdev": round(Shared.stdev(times) if len(times) > 1 else 0.0, 2),
                            "n_runs":              len(times),
                            "runs":               [round(t, 2) for t in times],
                        }
                        Shared.ok(f"{label} @ {res_label}: "
                           f"{results[short]['resolutions'][res_label]['sec_per_image_mean']:.1f}s/image")

                    artifact = None
                    if not last_images:
                        Shared.warn(f"{label} @ {res_label}: no images in ComfyUI history response — skipping save")
                    else:
                        img  = last_images[0]
                        dest = img_dir / f"{short}_{res_label}.png"
                        saved = ImageBenchmark.save_representative_image(
                            img, dest, comfyui_dir,
                        )

                        if saved:
                            artifact = ContentStore(img_dir / ".artifacts").put_file(
                                dest, "image/png",
                            ).to_dict()

                    if journal:
                        if not times and resolution_status == "ok":
                            resolution_status = "failed"
                        journal.record_resolution(
                            model, w, h, times, config.N_RUNS, resolution_status,
                            attempt_number=attempt_number, artifact=artifact,
                        )
                        segment_work += len(times)
                        results = journal.export()

                    if model_timed_out:
                        Shared.warn(f"{label}: timed out — moving to next model")
                        break

            finally:
                if telemetry_active and telemetry:
                    memory = telemetry.finish_case()
                    if journal:
                        power = getattr(telemetry, "last_power", None)
                        journal.record_model_evidence(
                            model, memory,
                            add_power_efficiency(
                                power, "images_per_joule", segment_work,
                            ),
                        )
                        results = journal.export()
                    elif isinstance(results.get(short), dict):
                        results[short]["memory"] = memory
                        if (power := getattr(telemetry, "last_power", None)) is not None:
                            work = sum(
                                resolution.get("n_runs", 0)
                                for resolution in results[short].get("resolutions", {}).values()
                                if isinstance(resolution, dict)
                            )
                            results[short]["power"] = add_power_efficiency(
                                power, "images_per_joule", work,
                            )
                if save_fn and not journal:
                    save_fn(results)
                Shared.log(f"Unloading {label} from VRAM ...")
                ImageBenchmark.comfyui_free_models()
                emit_model_finished("img", label, results.get(short), model_id=short)

        if journal:
            journal.finish()
            return journal.export()
        return results

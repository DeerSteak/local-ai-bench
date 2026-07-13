"""image_benchmark.py — ComfyUI-driven image generation benchmark."""

import time
from pathlib import Path

import requests

import config
from shared import Shared


class ImageBenchmark:
    @staticmethod
    def comfyui_free_models(timeout: int = 10) -> None:
        """Unload whatever checkpoint(s) ComfyUI currently has resident in memory.

        ComfyUI's own automatic model-swap-on-load is the only thing that would
        otherwise free a previous checkpoint, and on the MPS backend its free-VRAM
        detection is unreliable — models can stay resident far longer than on
        CUDA. Call this between models so each one starts from a clean memory
        state instead of stacking on top of whatever the last one left behind.
        """
        try:
            requests.post(f"{config.COMFYUI_URL}/free",
                          json={"unload_models": True, "free_memory": True},
                          timeout=timeout)
        except Exception as e:
            Shared.warn(f"Could not unload ComfyUI models: {e}")

    @staticmethod
    def comfyui_interrupt_and_clear(timeout: int = 10, confirm_timeout: int = 15) -> None:
        """Stop ComfyUI's currently running job and drop anything still queued.

        ComfyUI executes one job at a time. If we give up on a job client-side
        after a timeout without telling the server, it (or whatever we submit
        next) keeps occupying that single execution slot — every subsequent
        submission queues silently behind it and can time out in turn without
        ever actually starting. Call this right after a timeout so the next
        submission starts from a clean queue.

        /interrupt and /queue clear only signal ComfyUI — they return before the
        running job has actually unwound, so we poll /queue afterward until both
        queue_running and queue_pending are actually empty (or we give up and warn).
        """
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
        """
        Flux.1 txt2img workflow.

        The BFL flux1-schnell/dev .safetensors files are transformer-only (no CLIP,
        no VAE), so we load CLIP and VAE via separate nodes rather than relying on
        CheckpointLoaderSimple output slots 1 and 2 (which would be None).
        """
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
        """
        Flux.2-dev txt2img workflow.

        Flux.2 uses a Mistral-3-24B text encoder (loaded via a single CLIPLoader,
        type "flux2") instead of the T5-XXL + CLIP-L pair used by Flux.1/SD3, and
        a dedicated flux2-vae.safetensors — reusing Flux.1's DualCLIPLoader/VAE
        here silently produces a text-embedding-dimension mismatch deep in the
        transformer (txt_in linear layer) rather than a clear error.
        """
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
    def build_sd3_workflow(checkpoint, width, height, steps, cfg,
                           sampler, scheduler, seed, prompt, filename_prefix="bench_sd3"):
        """
        SD3.5 Large txt2img workflow for ComfyUI.

        sd3.5_large.safetensors contains the UNet and VAE but NOT the text encoders.
        clip_l.safetensors, clip_g.safetensors, and t5xxl_fp16.safetensors must be
        present in ComfyUI/models/clip/ (downloaded by setup_check.py).
        SD3 uses 16-channel latents — EmptySD3LatentImage is required.
        """
        return {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": checkpoint}},
            "2": {"class_type": "TripleCLIPLoader",
                  "inputs": {
                      "clip_name1": "clip_l.safetensors",
                      "clip_name2": "clip_g.safetensors",
                      "clip_name3": "t5xxl_fp16.safetensors",
                      "type": "sd3",
                  }},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "", "clip": ["2", 0]}},
            "5": {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}},
            "6": {"class_type": "KSampler",
                  "inputs": {
                      "model":          ["1", 0],
                      "positive":       ["3", 0],
                      "negative":       ["4", 0],
                      "latent_image":   ["5", 0],
                      "seed":           seed,
                      "steps":          steps,
                      "cfg":            cfg,
                      "sampler_name":   sampler,
                      "scheduler":      scheduler,
                      "denoise":        1.0,
                  }},
            "7": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
            "8": {"class_type": "SaveImage",
                  "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
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
    def build_workflow(workflow_t, checkpoint, width, height, steps, cfg,
                       sampler, scheduler, seed, prompt, filename_prefix):
        """Route to the right workflow builder for `workflow_t` (see models.py's
        "workflow" field). Unrecognized types fall through to the plain SDXL
        graph, which is the minimal loader→CLIP→KSampler→VAE shape that also
        works unchanged for SD1.5 (see models.py's IMAGE_MODELS comment)."""
        if workflow_t == "flux":
            builder = ImageBenchmark.build_flux_workflow
        elif workflow_t == "flux2":
            builder = ImageBenchmark.build_flux2_workflow
        elif workflow_t == "sd3":
            builder = ImageBenchmark.build_sd3_workflow
        else:
            builder = ImageBenchmark.build_sdxl_workflow
        return builder(checkpoint, width, height, steps, cfg, sampler, scheduler,
                       seed, prompt, filename_prefix=filename_prefix)

    @staticmethod
    def comfyui_submit(workflow: dict, timeout: int = 300) -> tuple[float, list[dict]]:  # pragma: no cover — submits to and polls a real ComfyUI server
        """Submit a workflow to ComfyUI, poll until done.

        Returns (elapsed_sec, images) where images is a list of
        {"filename": str, "subfolder": str, "type": str} dicts from all output nodes.
        """
        # A prior job (from an earlier model/test) can be left running or queued
        # if its own timeout handling didn't fully clear it — e.g. the interrupt
        # or queue-clear request itself failed. Check first so a stuck job never
        # silently eats our execution slot and causes a fresh, unrelated timeout.
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

                # Check for errors first
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

    def run(self, image_models, resolutions, seed, prompt,  # pragma: no cover — orchestrates real ComfyUI runs
            comfyui_dir, timeout=None, save_fn=None, images_dir=None):
        if timeout is None:
            timeout = config.RUN_TIMEOUT
        if images_dir is None:
            images_dir = config.RESULTS_DIR / "images"
        results = {}
        Shared.section("Image Generation via ComfyUI")

        checkpoints_dir = comfyui_dir / "models" / "checkpoints"

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

            try:
                # Skip if checkpoint not present
                ckpt_path = checkpoints_dir / checkpoint
                if not ckpt_path.exists():
                    Shared.warn(f"{label}: checkpoint not found at {ckpt_path} — skipping")
                    Shared.log(f"Download and place at: {ckpt_path}")
                    continue

                Shared.ok(f"{label}: checkpoint found ({ckpt_path.stat().st_size / (1024**3):.1f} GB)")
                results[short] = {"label": label, "checkpoint": checkpoint,
                                  "steps": steps, "resolutions": {}}

                # Warmup: one generation at the smallest resolution to trigger Metal/CUDA
                # shader compilation before timing starts.
                w0, h0 = model_resolutions[0]
                Shared.log(f"{label}: warmup run ({w0}x{h0}, timeout: {timeout}s) ...")
                warmup_ok = True
                # Use a seed outside the measured runs' range (seed .. seed+N_RUNS-1) so
                # this warmup can't hit the same ComfyUI node cache as measured run 1,
                # which would otherwise return near-instantly instead of regenerating.
                warmup_seed = seed - 1
                try:
                    wf = ImageBenchmark.build_workflow(workflow_t, checkpoint, w0, h0, steps, cfg,
                                                       sampler, scheduler, warmup_seed, prompt,
                                                       filename_prefix=f"{short}_warmup")
                    ImageBenchmark.comfyui_submit(wf, timeout=timeout)
                    Shared.ok(f"{label}: warmup done")
                except Exception as e:
                    Shared.warn(f"{label}: warmup failed ({e}) — skipping")
                    if not Shared.comfyui_available():
                        Shared.warn(f"ComfyUI appears to have crashed — last output:\n{Shared.tail_comfyui_log()}")
                    warmup_ok = False

                if not warmup_ok:
                    continue

                img_dir = images_dir

                model_timed_out = False
                for (w, h) in model_resolutions:
                    res_label = f"{w}x{h}"
                    Shared.log(f"{label} @ {res_label} — {config.N_RUNS} runs ...")
                    times = []
                    last_images: list[dict] = []

                    for run_i in range(config.N_RUNS):
                        try:
                            prefix = f"{short}_{res_label}_run{run_i + 1}"
                            # Vary the seed per run — an identical seed/workflow lets
                            # ComfyUI cache every node, so repeat runs return near-
                            # instantly instead of re-running generation.
                            run_seed = seed + run_i
                            wf = ImageBenchmark.build_workflow(workflow_t, checkpoint, w, h, steps, cfg,
                                                               sampler, scheduler, run_seed, prompt,
                                                               filename_prefix=prefix)

                            elapsed, images = ImageBenchmark.comfyui_submit(wf, timeout=timeout)
                            times.append(elapsed)
                            last_images = images
                            print(f"    run {run_i+1}/{config.N_RUNS}: {elapsed:.1f}s")
                        except TimeoutError:
                            Shared.err(f"Run {run_i+1} timed out — skipping {label}")
                            model_timed_out = True
                            results[short]["timed_out"] = res_label
                            break
                        except Exception as e:
                            Shared.err(f"Run {run_i+1} failed: {e}")

                    if times:
                        results[short]["resolutions"][res_label] = {
                            "sec_per_image_mean":  round(Shared.mean(times),  2),
                            "sec_per_image_stdev": round(Shared.stdev(times) if len(times) > 1 else 0.0, 2),
                            "n_runs":              len(times),
                            "runs":               [round(t, 2) for t in times],
                        }
                        Shared.ok(f"{label} @ {res_label}: "
                           f"{results[short]['resolutions'][res_label]['sec_per_image_mean']:.1f}s/image")

                    if not last_images:
                        Shared.warn(f"{label} @ {res_label}: no images in ComfyUI history response — skipping save")
                    else:
                        img  = last_images[0]
                        dest = img_dir / f"{short}_{res_label}.png"
                        saved = False
                        try:
                            ImageBenchmark.save_comfyui_image(img, dest)
                            Shared.ok(f"Saved image → {dest.relative_to(config.SCRIPT_DIR)}")
                            saved = True
                        except Exception as e:
                            Shared.warn(f"HTTP image fetch failed ({e}) — trying direct file copy")
                        if not saved:
                            # Fallback: copy directly from ComfyUI's output directory
                            subfolder = img.get("subfolder", "")
                            src = (comfyui_dir / "output" / subfolder / img["filename"]
                                   if subfolder else comfyui_dir / "output" / img["filename"])
                            try:
                                import shutil
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src, dest)
                                Shared.ok(f"Saved image (file copy) → {dest.relative_to(config.SCRIPT_DIR)}")
                            except Exception as e:
                                Shared.warn(f"Could not save image: {e}")

                    if model_timed_out:
                        Shared.warn(f"{label}: timed out — moving to next model")
                        break

            finally:
                if save_fn:
                    save_fn(results)
                Shared.log(f"Unloading {label} from VRAM ...")
                ImageBenchmark.comfyui_free_models()

        return results

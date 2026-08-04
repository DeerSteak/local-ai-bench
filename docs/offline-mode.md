# Offline Mode

Offline mode is an explicit benchmark execution setting available in the GUI and as `--offline`. It allows loopback and local Unix-socket communication required by llama.cpp and ComfyUI, blocks non-loopback connections made by the Python benchmark and supervised Python runners, and propagates common offline/telemetry-disabled environment settings to managed child processes.

The blocked targets include public internet hosts and private LAN addresses; `127.0.0.0/8`, `::1`, `localhost`, and local Unix sockets remain available. A blocked Python connection raises a specific offline-network error instead of silently falling back online. The resolved setting is identity-bearing in the run plan, portable presets/projects, result bundle, and decision report.

```bash
bash run_bench.sh --offline
```

Offline mode is for execution after setup has already installed runtimes, models, checkpoints, and question banks. Setup and model acquisition are separate consented workflows and may require network access. A ComfyUI instance on another machine is intentionally unavailable because it is not loopback.

## Enforcement boundary

The application guards TCP `connect`/`connect_ex` and UDP/message `sendto`/`sendmsg` on sockets created after offline mode activates, and sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`, and `WANDB_DISABLED` for inherited runtimes. A module that retained the original socket class before activation or an arbitrary native binary is outside what cross-platform Python can reliably firewall, so startup applies the policy before benchmark workload imports and stable-release qualification still requires packet-capture or equivalent OS-level observation on the supported hardware matrix. Until that qualification is recorded, offline mode is a strong application control rather than a claim that every process has been independently proven silent.

[← Security and Privacy](security-and-privacy.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)

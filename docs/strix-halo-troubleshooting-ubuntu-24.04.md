[← Back to documentation](README.md)

# Strix Halo Troubleshooting: Ubuntu 24.04

This runbook covers the native-Linux qualification path for AMD Ryzen AI Max+ 395 systems with Radeon 8060S graphics (`gfx1151`) on Ubuntu 24.04. It records the recovery validated on a clean Ubuntu installation after a generic-kernel downgrade removed GPU and 10GbE device support and an earlier shared ROCm installer incorrectly installed the discrete-Radeon DKMS driver.

## Required installation boundary

Strix Halo and a discrete Radeon GPU do not use the same native ROCm driver installation:

| Hardware | Kernel driver | ROCm command |
|---|---|---|
| Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) | Ubuntu OEM inbox `amdgpu` | `amdgpu-install -y --usecase=rocm --no-dkms` |
| Discrete Radeon RX 9060 XT (`gfx1200`) | AMD discrete graphics/DKMS stack | `amdgpu-install -y --usecase=graphics,rocm` |

Do not install `amdgpu-dkms` on Strix Halo. AMD's Ryzen instructions require Ubuntu's OEM kernel and the inbox driver; the qualification launcher installs `linux-oem-24.04`, ROCm 7.2.1, and the user's `render` and `video` group memberships. See AMD's [Ryzen ROCm installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).

## Recognizing the wrong kernel or driver

The failed configuration had these symptoms:

- `uname -r` reported `6.17.0-14-generic` rather than an `-oem` kernel.
- `rocminfo` reported `ROCk module is NOT loaded, possibly no GPU devices`.
- `lspci -nnk -d 1002:` listed `Kernel modules: amdgpu` but no `Kernel driver in use: amdgpu` for device `1002:1586`.
- `/dev/kfd` did not exist.
- The system's 10GbE interface disappeared at the same time, while Wi-Fi remained usable. This is a useful indication that the kernel change affected more than ROCm, but the Ethernet controller must still be verified independently after recovery.

The generic and OEM kernels can have the same major/minor version while carrying different device enablement. `6.17.0-14-generic` and the recovered `6.17.0-1032-oem` are not interchangeable qualification environments.

## Install and boot the OEM kernel

The updated qualification launcher installs the OEM kernel automatically when required. To repair the kernel before running qualification, use:

```bash
sudo apt update
sudo apt install linux-oem-24.04
ls -1 /boot/vmlinuz-*-oem
sudo reboot
```

Do not reboot until the `ls` command shows an installed OEM kernel. After reboot, verify that the selected kernel ends in `-oem`:

```bash
uname -r
```

If an OEM kernel is installed but GRUB still selects a generic kernel, retain both kernels and select the exact OEM entry under **Advanced options for Ubuntu** before removing or changing any kernel packages. Confirm a successful OEM boot before making it the persistent default.

## Recover from the old DKMS installation

An affected pre-fix qualification log contained `amdgpu-dkms is already the newest version` and showed DKMS building AMD modules for both the generic and OEM kernels. That proves the discrete-Radeon path ran on the Halo. Rebooting alone does not correct this contamination because the out-of-tree module under `updates/dkms` can take precedence over the OEM inbox module.

Use commit `cd21b58` or newer and run the normal target:

```bash
./run_qualification.sh ryzen-ai-halo-llamacpp-rocm
```

The managed recovery performs the equivalent of:

```bash
sudo dpkg --purge amdgpu-dkms
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo amdgpu-install -y --usecase=rocm --no-dkms
sudo usermod -aG render,video "$USER"
```

Prefer the qualification launcher so the exact managed path and its output are retained. Do not run `apt autoremove` while recovering; review its proposed removals only after the GPU works, because the previous driver installation may have marked firmware and kernel-support packages as automatically installed.

Purging `amdgpu-dkms` removes the active module from the running kernel. The setup attempt can therefore finish package recovery and still report that `rocminfo` cannot see a GPU. That is an expected reboot boundary, not evidence that the corrected installation failed:

```bash
sudo reboot
```

Group membership also takes effect only in a new login session, so `rocminfo` may require elevated access until that reboot. Qualification must run as the normal user and must not depend on `sudo rocminfo`.

## Verify the recovered driver

After reboot, run:

```bash
uname -r
modinfo -F filename amdgpu
lspci -nnk -d 1002:
ls -l /dev/kfd
rocminfo | grep gfx1151
```

A healthy recovery has all of these properties:

- `uname -r` ends in `-oem`.
- `modinfo` resolves `amdgpu` from the OEM kernel's normal `kernel/drivers/...` tree, not `updates/dkms`.
- The Radeon 8060S entry reports `Kernel driver in use: amdgpu`.
- `/dev/kfd` exists and is accessible to the normal user.
- `rocminfo` reports `gfx1151` without `sudo`.

If the inbox driver still does not bind after reboot, collect the first kernel-level failure instead of reinstalling ROCm again:

```bash
sudo modprobe amdgpu
sudo dmesg | grep -iE 'amdgpu|firmware|gfx1151' | tail -n 150
```

For Ethernet that remains unavailable after the OEM boot, collect its device and driver identity separately:

```bash
lspci -nnk | grep -A3 -i ethernet
ip -br link
sudo dmesg | grep -iE 'ethernet|firmware|network|atlantic|ixgbe|ice' | tail -n 100
```

Do not change NetworkManager configuration or install a third-party Ethernet driver until these commands establish whether the OEM kernel already exposes the controller.

## Resume qualification

Once `rocminfo` reports `gfx1151`, run the normal smallest-model qualification:

```bash
./run_qualification.sh ryzen-ai-halo-llamacpp-rocm
```

The launcher reuses the installed ROCm stack, installs the project-managed llama.cpp toolset, ComfyUI environment, and qualifying models, and then runs the ordinary benchmark. After llama.cpp qualification, the separate managed vLLM target is:

```bash
./run_qualification.sh ryzen-ai-halo-vllm-rocm
```

Setup output is retained before a benchmark journal exists. Every attempt appends terminal output to `qualification-evidence/TARGET/setup.log` and writes the latest attempt state to `qualification-evidence/TARGET/setup-status.json`. A setup interruption leaves the status as `running`; a completed setup attempt records `passed` or `failed`, its exit code, timestamps, and the log name.

An initial `Temporary failure in name resolution` while fetching Ubuntu or `repo.radeon.com` metadata is a network/DNS failure, not a ROCm compatibility verdict. Restore connectivity and rerun the same command; the append-only setup log preserves both attempts.

## Qualification identity

Record the exact kernel, ROCm version, accelerator identity, runtime version, suite version, and result artifacts. A successful Halo result qualifies only the tested Ubuntu 24.04 OEM-kernel configuration; it does not imply support for the generic kernel, the vendor's preinstalled image, another ROCm release, or the discrete-Radeon DKMS path.

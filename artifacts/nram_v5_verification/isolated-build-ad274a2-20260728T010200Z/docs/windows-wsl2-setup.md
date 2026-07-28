# Windows 11 + WSL2 + Docker Desktop GPU Setup (RTX 4090)

This guide sets up GPU-accelerated Docker on Windows 11 for NeuralAccessPsyche. The target hardware is an **NVIDIA RTX 4090** (24 GB VRAM), but the steps apply to any CUDA-capable NVIDIA GPU.

## Key principle: GPU access comes from the Windows driver

> **Do NOT install a Linux NVIDIA display driver inside WSL2.**
>
> GPU access under WSL2 is provided by the **Windows NVIDIA driver** through a paravirtualized interface. The only thing that needs to be installed *inside* the container (or WSL) is the **CUDA user-mode library**, which the official `nvidia/cuda` images already provide. Installing a full Linux NVIDIA display driver in WSL will conflict with the Windows driver and break GPU access.

In short:

- **On Windows (host):** install the regular NVIDIA GeForce / Studio driver.
- **Inside WSL2 / containers:** nothing driver-level to install — CUDA libs ship in the images.

## Prerequisites

| Item | Value verified in baseline |
|------|---------------------------|
| Windows | 11 (10.0.26300.8935) |
| WSL | 2.7.10.0, kernel 6.18.33.2-2 |
| Distro | Ubuntu 26.04 LTS |
| Docker Desktop | 4.83.0 |
| NVIDIA driver (Windows) | 591.86 |
| CUDA (nvidia-smi) | 13.1 |
| GPU | RTX 4090, 23028 MiB VRAM |

See [`runtime-baseline.md`](runtime-baseline.md) for the exact recorded environment.

## Step 1 — Install WSL2

From an **elevated PowerShell**:

```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

Reboot, then launch Ubuntu and create your UNIX user.

## Step 2 — Install the Windows NVIDIA driver

Download and install the current **GeForce Game Ready / Studio driver** for the RTX 4090 from [nvidia.com](https://www.nvidia.com/Download/index.aspx). This single Windows driver is what exposes the GPU to WSL2.

Verify from PowerShell:

```powershell
nvidia-smi
```

You should see the RTX 4090 with the driver and CUDA version.

## Step 3 — Install Docker Desktop + WSL2 integration

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. Enable the **WSL 2 based engine** (Settings → General).
3. Enable Ubuntu integration: **Settings → Resources → WSL Integration → enable Ubuntu**.

> **Note from baseline:** the `docker` CLI is **not available inside WSL2 Ubuntu** by default (`docker: command not found`). That is expected when Docker runs via Docker Desktop. Run all Docker commands from **Windows PowerShell**, or enable the WSL integration toggle above.

## Step 4 — Validate GPU passthrough into a container

Run the NVIDIA CUDA base image and confirm `nvidia-smi` works **inside the container**:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Expected result: the RTX 4090 is listed, matching the host driver/CUDA versions. This was **PASSED** in the recorded baseline (image digest `sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2`).

If this fails, GPU passthrough is broken and SGLang will not start with `--gpus all`. Fix the driver / WSL integration before continuing.

## Step 5 — The local GGUF model (mounted read-only, not downloaded)

The model is **not** downloaded by the container. It is a local file on the Windows `E:` drive, mounted **read-only** into the SGLang container via its WSL path:

| Item | Value |
|------|-------|
| File | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` |
| Windows path | `E:\_MODELS\huggingface\hub\DeepSeek-R1-Distill-Qwen-7B-GGUF\` |
| WSL mount path | `/mnt/e/_MODELS/huggingface/hub/DeepSeek-R1-Distill-Qwen-7B-GGUF/` |
| Size | 4.36 GB |

Windows drives are auto-mounted under `/mnt/<drive-letter>` in WSL2, so `E:\` appears as `/mnt/e`. The compose file mounts this directory into the SGLang container as a **read-only** volume — the container never writes to or downloads the model.

```yaml
volumes:
  - /mnt/e/_MODELS/huggingface/hub/DeepSeek-R1-Distill-Qwen-7B-GGUF:/models:ro
```

The `:ro` flag enforces read-only access.

## Step 6 — Start the stack

```bash
docker compose up --build
```

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `nvidia-smi` fails in container | Windows NVIDIA driver missing/outdated, or WSL integration disabled |
| `docker: command not found` inside WSL | Run Docker from PowerShell, or enable WSL Integration toggle |
| Model not found at `/models` | Mount path mismatch — confirm the WSL `/mnt/e` path exists |
| Out-of-memory on start | Too little free VRAM — close desktop apps (baseline had ~10 GB free with ~12 GB in use) |

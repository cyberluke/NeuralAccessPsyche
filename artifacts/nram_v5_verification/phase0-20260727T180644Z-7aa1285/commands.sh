#!/usr/bin/env bash
# Phase-0 command ledger. Run from d:/_SATIN_AI/NeuralAccessPsyche.
# The original execution host was Windows 11 with PowerShell 7; docker exec
# commands target Linux containers. Secrets are not included.

git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --porcelain=v2 --branch --untracked-files=all
git diff --stat
git diff --cached --stat
git show -s --format='%H%n%ci%n%s' 6b92283
git merge-base --is-ancestor 6b92283 HEAD

wsl.exe --version
wsl.exe --status
wsl.exe --list --verbose
docker version
docker compose version
docker ps --no-trunc
docker compose ps --all --format json
docker compose images --format json
nvidia-smi
nvcc --version
python --version
uv --version

docker image inspect lmsysorg/sglang:latest
docker image inspect neuralaccesspsyche-nram-api:latest
docker inspect nram-sglang
docker inspect nram-api
docker exec nram-sglang ps -ww -eo pid,ppid,args
docker exec nram-sglang python3 -m sglang.launch_server --help
docker exec nram-sglang python3 -c 'import importlib.metadata as m; print(m.version("sglang"))'
docker exec nram-sglang nvidia-smi

docker exec nram-sglang sh -lc "cd /models && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum"
docker exec nram-sglang sh -lc "cd /models && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum"

# Collection with environment plugins disabled and only required async plugin loaded.
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run --isolated --frozen --with pytest --with pytest-asyncio --with dill \
  --with cloudpickle --with torch==2.13.0 python -m pytest tests \
  -p no:cacheprovider -p pytest_asyncio.plugin --tb=long -ra \
  --junitxml=artifacts/nram_v5_verification/phase0-20260727T180644Z-7aa1285/test-results.xml

npm test
docker compose config --quiet
docker compose config --images

curl -H 'Authorization: Bearer phase0-local' http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready

# Exact request JSON, responses, hashes, and timings are preserved in
# runtime-evidence.json and the Phase-0 intake report. The controlled forced-token
# request was sent from nram-api to http://sglang:30000/v1/chat/completions using
# nram_sglang.processor.NRAMLogitProcessor, token 11064, and all other model IDs masked.

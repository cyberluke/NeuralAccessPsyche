# PowerShell-host commands (executed from repository root)
python -m pytest --collect-only -q
python -m pytest tests/unit tests/contract tests/adversarial tests/integration -q
python -m pytest tests/gpu -q -s
python -m pytest tests -q --junitxml=<artifact-dir>/test-results.xml
python -m ruff check .
python -m mypy nram_sglang/processor.py core/steering/serialization.py core/engines/sglang_engine.py api/routes.py
docker compose config --quiet
docker compose build sglang nram-api
docker compose up -d --force-recreate --wait --wait-timeout 600 sglang
docker compose up -d --force-recreate --no-deps nram-api
python evaluation/nram_v5_paired.py --artifact-dir <artifact-dir> --max-prompts 2 --seeds 101,202 --max-tokens 24

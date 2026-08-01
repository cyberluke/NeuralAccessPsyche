"""Focused tests for default-off DExperts runtime isolation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.contracts.nram_runtime import normalize_nram_options
from core.steering.request_control_plane import (
    NRAMRequestConfig,
    NRAMRequestControlPlane,
    build_request_config_from_nram_opts,
)
from nram_sglang.processor import NRAMLogitProcessor
from utils.validators import validate_request


def test_default_runtime_has_no_dexperts_capability_or_adapter_initialization(monkeypatch):
    monkeypatch.delenv("NRAM_DEXPERTS_ENABLED", raising=False)

    def unexpected_runtime_lookup():
        raise AssertionError("DExperts adapter runtime lookup must be skipped")

    monkeypatch.setattr(
        "nram_sglang.processor._get_dexperts_runtime_class",
        unexpected_runtime_lookup,
    )
    NRAMLogitProcessor._dexperts_runtime = None
    NRAMLogitProcessor.initialize_dexperts(object(), object())

    capabilities = NRAMRequestControlPlane(NRAMRequestConfig()).get_active_capabilities()
    assert "dexperts" not in capabilities
    assert NRAMLogitProcessor._dexperts_runtime is None


def test_default_runtime_rejects_explicit_dexperts_request(monkeypatch):
    monkeypatch.delenv("NRAM_DEXPERTS_ENABLED", raising=False)
    request = SimpleNamespace(
        messages=[{"role": "user", "content": "hello"}],
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=1,
        response_format=None,
        nram={"dexperts": True},
    )

    with pytest.raises(HTTPException) as error:
        validate_request(request)

    assert error.value.status_code == 400
    assert error.value.detail["code"] == "DEXPERTS_DISABLED"


def test_enabled_flag_preserves_existing_dexperts_configuration_path(monkeypatch):
    monkeypatch.setenv("NRAM_DEXPERTS_ENABLED", "true")
    options = normalize_nram_options(
        {"dexperts": True, "dexperts_config": {"enabled": True}}
    )
    config = build_request_config_from_nram_opts(
        {"dexperts": {"enabled": True, "expert_name": "nontoxic"}}
    )

    assert options["dexperts"] is True
    assert config.dexperts.enabled is True
    assert config.dexperts.expert_name == "nontoxic"

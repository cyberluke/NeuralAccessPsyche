"""Headless Streamlit application construction gate."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_streamlit_console_constructs_without_exception():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).parents[2] / "streamlit_console.py")
    )
    app.run(timeout=30)
    assert not app.exception

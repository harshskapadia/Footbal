"""Smoke test for the Streamlit dashboard (runs the script headless, no browser)."""

from streamlit.testing.v1 import AppTest

from src.utils.config import REPO_ROOT


def test_dashboard_loads_without_errors() -> None:
    at = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Match analysis"
    assert at.radio[0].options == ["Upload a clip", "Clip in data/raw", "Previous results"]

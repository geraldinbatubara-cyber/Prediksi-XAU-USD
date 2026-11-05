"""Guard against accidentally retraining models during Streamlit UI reruns.

The dashboard training jobs are intentionally owned by GitHub Actions.
This is a source-level contract, so importing the Streamlit app is unnecessary.
"""
import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_does_not_build_or_save_snapshot_in_streamlit_runtime():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    disallowed = {"build_dashboard_snapshot", "save_dashboard_snapshot"}
    runtime_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (disallowed & runtime_calls)


def test_dashboard_still_checks_snapshot_freshness_and_reports_stale_data():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "dashboard_snapshot_is_current" in calls
    assert "Model tidak dilatih ulang di Streamlit Cloud" in source
    assert 'dashboard_snapshot["model_1"]' in source
    assert 'dashboard_snapshot["model_2"]' in source

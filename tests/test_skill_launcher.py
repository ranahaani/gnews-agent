"""skills/gnews/scripts/gnews_agent.py launcher script — error path + forwarding."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent.parent / "skills" / "gnews" / "scripts" / "gnews_agent.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("_gnews_skill_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_cli_prints_install_help(monkeypatch, capsys):
    module = _load_launcher()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    exit_code = module.main(["gnews_agent.py", "search", "X"])
    assert exit_code == 127
    err = capsys.readouterr().err
    assert "pip install gnews-agent" in err


def test_forwards_arguments(monkeypatch):
    module = _load_launcher()
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/local/bin/gnews-agent")
    seen: dict = {}

    def fake_call(cmd):
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr(module.subprocess, "call", fake_call)
    exit_code = module.main(["gnews_agent.py", "search", "GPT-5", "--limit", "3"])
    assert exit_code == 0
    assert seen["cmd"] == ["/usr/local/bin/gnews-agent", "search", "GPT-5", "--limit", "3"]

import json
from pathlib import Path

from comparison.standardized.run_all import Checkpoint, PHASES, command_plan, item_id, main


def test_atomic_checkpoint_resume_and_retry(tmp_path):
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    cp = Checkpoint(state, events)
    assert cp.start(item_id("train"))
    cp.finish(item_id("train"), "failed", "boom")
    assert not cp.start(item_id("train"), resume=True)
    assert cp.start(item_id("train"), resume=True, retry_failed=True)
    cp.finish(item_id("train"), "completed")
    assert not cp.start(item_id("train"), resume=True)
    assert json.loads(state.read_text())["items"]["train:default"]["status"] == "completed"
    assert len(events.read_text().splitlines()) == 4


def test_stale_running_is_restarted_and_recorded(tmp_path):
    cp = Checkpoint(tmp_path / "state.json", tmp_path / "events.jsonl")
    cp.state["items"]["cache:default"] = {"status": "running"}
    cp._save()
    assert cp.start("cache:default", resume=True)
    assert "stale_running_restarted" in cp.events_path.read_text()


def test_dry_run_is_side_effect_free(tmp_path, capsys):
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    assert main(["--dry-run", "--state", str(state), "--events", str(events)]) == 0
    assert not state.exists() and not events.exists()
    assert "preflight" in capsys.readouterr().out


def test_failed_phase_blocks_downstream(tmp_path, monkeypatch):
    from comparison.standardized import run_all
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    cp = Checkpoint(state, events)
    cp.finish("preflight:default", "failed", "missing input")
    calls = []
    monkeypatch.setattr(run_all.subprocess, "run", lambda *a, **kw: calls.append(a))
    assert main(["--execute", "--resume", "--state", str(state), "--events", str(events)]) == 1
    assert calls == []


def test_plan_builds_caches_then_audits_and_executes_all_explanation_sets():
    plan = command_plan()
    assert "comparison.standardized.build_caches" in plan["cache"][0]
    assert "--execute" in plan["cache"][0]
    assert "comparison.standardized.audit_caches" in plan["cache"][-1]
    assert len(plan["explain"]) == 6
    pairs = set()
    for command in plan["explain"]:
        assert "--dry-run" not in command
        pairs.add((command[command.index("--topology") + 1], command[command.index("--seed") + 1]))
    assert pairs == {(t, str(s)) for t in ("star", "cooccur") for s in (1234, 1235, 1236)}


def test_resume_tracks_individual_commands_not_whole_phase(tmp_path, monkeypatch):
    from comparison.standardized import run_all
    from types import SimpleNamespace
    plan = {phase: [] for phase in PHASES}
    plan["train"] = [["fixture", "one"], ["fixture", "two"], ["fixture", "three"]]
    monkeypatch.setattr(run_all, "command_plan", lambda **kwargs: plan)
    calls = []
    def execute(command, **kwargs):
        calls.append(command[-1])
        return SimpleNamespace(returncode=int(command[-1] == "two" and calls.count("two") == 1))
    monkeypatch.setattr(run_all.subprocess, "run", execute)
    args = ["--execute", "--state", str(tmp_path / "state.json"), "--events", str(tmp_path / "events.jsonl")]
    assert main(args) == 1
    assert calls == ["one", "two"]
    assert main(args + ["--resume"]) == 1
    assert calls == ["one", "two"]
    assert main(args + ["--resume", "--retry-failed"]) == 0
    assert calls == ["one", "two", "two", "three"]


def test_real_child_exit_stops_pipeline_and_resume_skips_completed(tmp_path, monkeypatch):
    import sys
    from comparison.standardized import run_all
    trace = tmp_path / "trace.txt"
    fail = tmp_path / "fail"
    fail.write_text("interrupt second step")
    def child(name, fails=False):
        code = (f"from pathlib import Path; p=Path({str(trace)!r}); "
                f"p.open('a').write({name!r}+'\\n'); "
                f"raise SystemExit(int(Path({str(fail)!r}).exists()) if {fails!r} else 0)")
        return [sys.executable, "-c", code]
    plan = {phase: [] for phase in PHASES}
    plan["train"] = [child("one"), child("two", True)]
    plan["explain"] = [child("explain")]
    monkeypatch.setattr(run_all, "command_plan", lambda **kwargs: plan)
    args = ["--execute", "--state", str(tmp_path / "state.json"), "--events", str(tmp_path / "events.jsonl")]
    assert main(args) == 1
    assert trace.read_text().splitlines() == ["one", "two"]
    assert main(args + ["--resume"]) == 1
    assert trace.read_text().splitlines() == ["one", "two"]
    fail.unlink()  # Temporary test sentinel, not a repository artifact.
    assert main(args + ["--resume", "--retry-failed"]) == 0
    assert trace.read_text().splitlines() == ["one", "two", "two", "explain"]


def test_commands_are_argv_lists_without_shell_tokens():
    plan = command_plan()
    assert tuple(plan) == PHASES
    assert all(isinstance(arg, str) for commands in plan.values() for cmd in commands for arg in cmd)
    assert all("shell=True" not in cmd for commands in plan.values() for cmd in commands)

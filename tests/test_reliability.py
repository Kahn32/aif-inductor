"""Failure-handling tests. Run: python tests/test_reliability.py"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aif_inductor import induce as induce_mod
from aif_inductor import llm as llm_mod
from aif_inductor.model_spec import InvalidModel, validate
from aif_inductor.tmaze import generate_trajectories, true_model


OK_BODY = {"model": "served/model", "provider": "TestHost",
           "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def fake_llm(responses, **kwargs):
    """OpenRouterLLM pointed at a local server replaying (status, body) pairs.
    The last response repeats. Returns the client, the server and received bodies."""
    received, queue = [], list(responses)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = queue.pop(0) if len(queue) > 1 else queue[0]
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    try:
        client = llm_mod.OpenRouterLLM("m", min_gap_s=0, retry_wait_s=0, rate_wait_s=0, **kwargs)
    finally:
        del os.environ["LLM_BASE_URL"]
    return client, server, received


def data():
    return generate_trajectories(3, seed=1), generate_trajectories(3, seed=2)


def test_empty_reply_is_a_failed_round():
    train, test = data()
    best, trace = induce_mod.induce(lambda s, u: None, train, test, rounds=1, verbose=False)
    assert best is None and len(trace) == 2
    assert all(not e["ok"] and "empty" in e["error"] for e in trace)


def test_scoring_crash_is_a_failed_round():
    train, test = data()
    original = induce_mod.score
    def broken(*args, **kwargs):
        raise ValueError("scoring broke")
    induce_mod.score = broken
    try:
        reply = llm_mod.MOCK_PROPOSALS[0]
        best, trace = induce_mod.induce(lambda s, u: reply, train, test, rounds=0, verbose=False)
    finally:
        induce_mod.score = original
    assert best is None and "scoring broke" in trace[0]["error"]


def test_huge_model_is_rejected():
    spec = true_model()
    spec["factors"] = [("location", 1000, True), ("context", 1000, False)]
    try:
        validate(spec)
    except InvalidModel as e:
        assert "too large" in str(e)
    else:
        raise AssertionError("huge model was accepted")


def test_candidate_code_cannot_see_api_key():
    os.environ["OPENROUTER_API_KEY"] = "secret-test-value"
    try:
        out = induce_mod.run_code(
            "import os\ndef build_model():\n"
            "    return {'seen': os.environ.get('OPENROUTER_API_KEY', 'absent')}\n")
    finally:
        del os.environ["OPENROUTER_API_KEY"]
    assert out["seen"] == "absent"


def test_run_logs_are_not_overwritten():
    from aif_inductor.runs import prepare_run_dir, run_tag
    assert run_tag("a/b:free", 2, 10, "cue_first") == "a_b:free_n10_cue_first_seed2"
    assert run_tag("m", 0, 5) != run_tag("m", 0, 10)
    with tempfile.TemporaryDirectory() as d:
        run = prepare_run_dir(Path(d) / "run")
        (run / "trace.jsonl").write_text("evidence")
        try:
            prepare_run_dir(run)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing logs were not protected")
        assert (run / "trace.jsonl").read_text() == "evidence"
        prepare_run_dir(run, force=True)
        assert not (run / "trace.jsonl").exists()


def test_every_http_attempt_is_counted():
    client, server, _ = fake_llm([(500, {}), (200, OK_BODY)])
    try:
        assert client("s", "u") == "hello"
    finally:
        server.shutdown()
    assert client.usage["attempts"] == 2 and client.usage["calls"] == 1


def test_retries_are_limited():
    client, server, _ = fake_llm([(500, {})])
    try:
        client("s", "u")
    except Exception:
        pass
    else:
        raise AssertionError("expected failure")
    finally:
        server.shutdown()
    assert client.usage["attempts"] == 3


def run_script(cwd, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/run.py"), "--mock", "--rounds", "3", "--n-train", "3",
         "--n-test", "3", "--eval-episodes", "2", *extra],
        cwd=cwd, capture_output=True, text=True, timeout=300)


def test_daily_cap_leaves_partial_summary():
    with tempfile.TemporaryDirectory() as d:
        r = run_script(d, "--mock-stop-after", "1")
        assert r.returncode == 0, r.stderr[-1500:]
        run = Path(d) / "runs/mock_n3_random_seed0"
        summary = json.loads((run / "summary.json").read_text())
        trace = [json.loads(x) for x in (run / "trace.jsonl").read_text().splitlines()]
    assert summary["status"] == "partial" and "limit" in summary["stop_reason"]
    assert summary["induced_final_test"] == summary["per_round"][-1]["final_test"]
    assert len(summary["prompt_hash"]) == 12 and len(summary["git"]["commit"]) == 40
    assert len(trace) == 2 and trace[-1]["stopped"] and "induced" in summary["results"]


def test_seed_is_sent_to_provider():
    client, server, received = fake_llm([(200, OK_BODY)], seed=7)
    try:
        client("s", "u")
    finally:
        server.shutdown()
    assert received[0]["seed"] == 7


def test_served_model_is_logged():
    client, server, _ = fake_llm([(200, {**OK_BODY, "choices": [
        {"message": {"content": llm_mod.MOCK_PROPOSALS[0]}, "finish_reason": "stop"}]})])
    train, test = data()
    try:
        _, trace = induce_mod.induce(client, train, test, rounds=0, verbose=False)
    finally:
        server.shutdown()
    assert trace[0]["served"] == {"model": "served/model", "provider": "TestHost",
                                  "finish_reason": "stop"}


def test_reasoning_settings_are_sent():
    client, server, received = fake_llm([(200, OK_BODY)], max_tokens=123, reasoning_effort="low")
    try:
        client("s", "u")
    finally:
        server.shutdown()
    assert received[0]["max_tokens"] == 123
    assert received[0]["reasoning"] == {"effort": "low"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)

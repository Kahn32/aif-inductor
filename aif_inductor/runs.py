"""Run naming and protection of saved run logs."""
from pathlib import Path


def run_tag(model, seed, n_train, demo_policy="random"):
    """Name that differs whenever the data setting differs."""
    return f"{model.replace('/', '_')}_n{n_train}_{demo_policy}_seed{seed}"


def prepare_run_dir(path, force=False):
    """Create a run directory. Refuse to replace existing logs unless forced."""
    path = Path(path)
    logs = [path / "trace.jsonl", path / "summary.json"]
    if any(p.exists() for p in logs):
        if not force:
            raise FileExistsError(f"{path} already has run logs. Use a new setting or --force.")
        for p in logs:
            p.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path

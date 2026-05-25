"""MLflow 封装 smoke test：写入本地临时后端，验证 run 能正常起停并记录参数。"""

from __future__ import annotations

import mlflow

from ivp.common.config import load_config
from ivp.common.tracking import log_metrics, start_run


def test_start_run_logs_params(tmp_path) -> None:
    """开启 run 应记录扁平化配置，并能写入一条指标。"""
    cfg = load_config(
        overrides=[
            f"tracking.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
            "tracking.experiment_name=test-exp",
        ]
    )
    with start_run(cfg, run_name="smoke") as run:
        assert run is not None
        log_metrics({"dummy_metric": 1.0})
        run_id = run.info.run_id

    logged = mlflow.get_run(run_id)
    assert logged.data.params["seed"] == "42"
    assert logged.data.metrics["dummy_metric"] == 1.0


def test_disabled_tracking_yields_none() -> None:
    """tracking.enabled=false 时应跳过 MLflow，yield None。"""
    cfg = load_config(overrides=["tracking.enabled=false"])
    with start_run(cfg) as run:
        assert run is None

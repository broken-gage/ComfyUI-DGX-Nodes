import time
from contextlib import contextmanager


def _format_metric_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _format_metrics(metrics):
    ordered_keys = []
    if "runtime_s" in metrics:
        ordered_keys.append("runtime_s")
    if "status" in metrics:
        ordered_keys.append("status")
    ordered_keys.extend(key for key in metrics.keys() if key not in {"runtime_s", "status"})

    parts = []
    for key in ordered_keys:
        value = metrics.get(key)
        if value is None:
            continue
        if key == "runtime_s":
            parts.append(f"{key}={value:.3f}")
            continue
        parts.append(f"{key}={_format_metric_value(value)}")
    return ", ".join(parts)


@contextmanager
def node_timer(logger, node_name, **initial_metrics):
    start = time.perf_counter()
    metrics = dict(initial_metrics)
    try:
        yield metrics
    except Exception:
        metrics["status"] = "error"
        metrics["runtime_s"] = time.perf_counter() - start
        logger.exception("%s: %s", node_name, _format_metrics(metrics))
        raise
    else:
        metrics.setdefault("status", "ok")
        metrics["runtime_s"] = time.perf_counter() - start
        logger.info("%s: %s", node_name, _format_metrics(metrics))

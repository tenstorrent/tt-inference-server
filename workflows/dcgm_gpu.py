import requests, re, time, csv, pathlib, datetime
from collections import defaultdict
from typing import Dict, Optional

URL  = "http://localhost:9400/metrics"
# match metric name, optional {label,..} and numeric value.
PAT  = re.compile(r'^(DCGM_[^ {]+)(?:\{([^}]*)\})?\s+([+-]?[0-9.eE]+)$')
# will hold the set of metric names seen so far — filled after first sample
METRIC_KEYS = []

PERIOD = 0.050          # 50 ms  (seconds)
OUT    = pathlib.Path("gpu_metrics_50ms.csv")

# ────────────────── CSV writer ──────────────────
# Header will be written lazily after we know which metrics exist.
def write_csv(utc_ts, t, dt, gpu_id, model_name, metrics):
    """Append one row to CSV, creating file/header on first call."""
    global METRIC_KEYS

    if not METRIC_KEYS:
        # Capture the column order on first appearance
        METRIC_KEYS = sorted(metrics.keys())
        header = "utc,t,dt,gpu,modelName," + ",".join(METRIC_KEYS) + "\n"
        OUT.write_text(header)

    row = [utc_ts, f"{t:.6f}", f"{dt*1000:.2f}", gpu_id or "", model_name or ""]
    row += [str(metrics.get(k, "")) for k in METRIC_KEYS]
    with OUT.open("a") as fp:
        fp.write(",".join(row) + "\n")
# ─────────────────────────────────────────────────────────

last_t = time.perf_counter()          # initial timestamp
print("Press Ctrl-C to stop…")
while True:
    start = time.perf_counter()

    # Collect metrics for every GPU in this sample window
    by_gpu: Dict[str, Dict[str, float]] = defaultdict(dict)
    model_names: Dict[str, Optional[str]] = {}

    for line in requests.get(URL, timeout=1).text.splitlines():
        if line.startswith("#") or not line:
            continue  # skip HELP/TYPE comments and blank lines
        m = PAT.match(line)
        if m:
            metric, label_str, val = m.groups()

            labels = {}
            if label_str:
                labels = dict(re.findall(r'([^=,]+)="([^"]*)"', label_str))

            gpu = labels.get("gpu", "unknown")
            by_gpu[gpu][metric] = float(val)

            if gpu not in model_names:
                model_names[gpu] = labels.get("modelName")

    # --- show one line of output ---
    dt = start - last_t
    last_t = start
    utc_ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds") + "Z"

    # show only a few key metrics to keep the console output readable
    sample_keys = (
        "DCGM_FI_DEV_SM_CLOCK",
        "DCGM_FI_DEV_MEM_CLOCK",
        "DCGM_FI_DEV_GPU_TEMP",
        "DCGM_FI_DEV_POWER_USAGE",
    )

    # Emit one line per GPU
    for gpu_id in sorted(by_gpu.keys()):
        metrics = by_gpu[gpu_id]
        sample_str = " | ".join(
            f"{k.split('_')[-2]}:{metrics.get(k,'-')}" for k in sample_keys if k in metrics
        )
        print(
            f"{utc_ts} | dt = {dt*1000:6.1f} ms | gpu:{gpu_id} | "
            f"model:{model_names.get(gpu_id)} | {sample_str}"
        )

        # comment out the next line if you don't want the file
        write_csv(utc_ts, start, dt, gpu_id, model_names.get(gpu_id), metrics)

    # --- sleep the remaining time in the 50 ms period ---
    elapsed = time.perf_counter() - start
    time.sleep(max(0, PERIOD - elapsed))
import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "benchmark"
WORKER_DIR = "worker_metrics"
REQUEST_LOGS = "requests_*.csv"
SECURE_SESSIONS = "secure_sessions.csv"
SCHEDULE_LAG = "schedule_lag"
SECURE_POLL = "secure/session"

STEP_SECONDS = 2
MAX_POINTS = 10_000
BIN_SECONDS = 10

SERVICE = "container_label_com_docker_compose_service"
COMPOSE_FILTER = f'{SERVICE}!=""'
QUERIES = {
    "requests": "sum by (instance, handler, status) (rate(http_requests_total[10s]))",
    "server_latency_p95": "histogram_quantile(0.95, sum by (le, handler) "
                          "(rate(http_request_duration_seconds_bucket[10s])))",
    "in_progress": "sum by (instance) (http_requests_in_progress)",
    "ratio_429": 'sum by (handler) (rate(http_requests_total{status="429"}[30s])) '
                 "/ sum by (handler) (rate(http_requests_total[30s]))",
    "container_cpu": f"sum by ({SERVICE}) (rate(container_cpu_usage_seconds_total{{{COMPOSE_FILTER}}}[10s]))",
    "container_memory": f"sum by ({SERVICE}) (container_memory_working_set_bytes{{{COMPOSE_FILTER}}})",
    "container_net_tx": f"sum by ({SERVICE}) "
                        f"(rate(container_network_transmit_bytes_total{{{COMPOSE_FILTER}}}[10s]))",
    "host_cpu": '1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[10s]))',
    "queue_depth": 'redis_key_size{instance_role="broker", key=~"light|heavy"}',
    "task_rate": "sum by (name, hostname) (rate(celery_task_succeeded_total[30s]))",
    "task_failed": "sum by (name, hostname) (rate(celery_task_failed_total[30s]))",
    "task_runtime_p95": "histogram_quantile(0.95, sum by (le, name) (rate(celery_task_runtime_bucket[30s])))",
    "pg_connections": "sum by (state) (pg_stat_activity_count)",
    "pg_commits": 'rate(pg_stat_database_xact_commit{datname="somasafe"}[10s])',
    "redis_ops": "sum by (instance_role) (rate(redis_commands_processed_total[10s]))",
    "upstreams_healthy": "caddy_reverse_proxy_upstreams_healthy",
}
AGGREGATION_TASK = "federated_aggregation"
STAGED_TASKS = {"federated_aggregation": ("cohort", "submissions", "aggregated"),
                "secure_session_sum": ("members", "members", "summed")}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#898781"
SURFACE = "#fcfcfb"
SERVICES = ("caddy", "fastapi-1", "fastapi-2", "celery-1", "celery-2", "postgres", "redis-auth", "redis-broker")
SERVICE_COLORS = dict(zip(SERVICES, PALETTE))
STATUS_COLORS = {"summed": "#0ca30c", "failed": "#d03b3b", "in flight": MUTED}

plt.switch_backend("Agg")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
    "axes.titlesize": 10, "axes.titlelocation": "left", "font.size": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 8,
})


stage_marks: list[float] = []


def utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def query_range(prometheus: str, query: str, start: float, end: float) -> pd.DataFrame:
    rows = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + STEP_SECONDS * MAX_POINTS, end)
        params = urllib.parse.urlencode({"query": query, "start": cursor, "end": stop, "step": STEP_SECONDS})
        with urllib.request.urlopen(f"{prometheus}/api/v1/query_range?{params}", timeout=120) as resp:
            body = json.load(resp)
        if body["status"] != "success":
            raise RuntimeError(f"query failed: {query}: {body.get('error')}")
        for series in body["data"]["result"]:
            labels = {key: value for key, value in series["metric"].items() if key != "__name__"}
            rows.extend({"timestamp": float(ts), **labels, "value": float(value)} for ts, value in series["values"])
        cursor = stop + STEP_SECONDS
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "value"])


def cut_worker_metrics(run_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    parts = defaultdict(list)
    for path in sorted((run_dir / WORKER_DIR).glob("*/*.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "process", path.parent.name)
        parts[path.stem.rsplit(".", 1)[-1]].append(frame)
    tasks = {}
    for task, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True)
        started = pd.to_datetime(frame["started_at"], utc=True, format="ISO8601")
        frame = frame[(started >= start) & (started <= end)].sort_values("started_at")
        frame.to_csv(run_dir / f"worker_{task}.csv", index=False)
        tasks[task] = frame
    return tasks


def wide(frame: pd.DataFrame, start: float, labels: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if labels is None:
        labels = [column for column in frame.columns
                  if column not in ("timestamp", "value") and frame[column].nunique(dropna=False) > 1]
    keys = (frame[labels].astype(str).agg(" ".join, axis=1).str.replace("worker.tasks.", "", regex=False)
            if labels else pd.Series("value", index=frame.index))
    table = frame.assign(key=keys).groupby(["timestamp", "key"])["value"].sum(min_count=1).unstack("key")
    table.index = (table.index - start) / 60
    return table


def mark_stages(ax) -> None:
    for minute in stage_marks:
        ax.axvline(minute, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=0)


def no_data(ax) -> None:
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)


def plot_lines(ax, table: pd.DataFrame, colors: dict[str, str] | None = None) -> None:
    if table.empty:
        no_data(ax)
        return
    named = colors is not None
    if colors is None:
        shown = sorted(table.max().sort_values(ascending=False).index[:len(PALETTE)])
        if len(shown) < table.shape[1]:
            ax.set_title(f"{ax.get_title(loc='left')} (top {len(shown)} of {table.shape[1]})")
        table = table[shown]
        colors = dict(zip(shown, PALETTE))
    for key in table.columns:
        ax.plot(table.index, table[key], color=colors.get(key, MUTED), label=key)
    if named or table.shape[1] > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))


def time_axes(rows: int, title: str, duration: float, height: float = 2.2):
    fig, axes = plt.subplots(rows, 1, figsize=(10, height * rows), sharex=True, layout="constrained", squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    axes = axes[:, 0]
    for ax in axes:
        mark_stages(ax)
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("minutes since start")
    return fig, axes


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_client(requests: pd.DataFrame, start: float, duration: float, path: Path) -> None:
    frame = requests.assign(bin=(requests["timestamp"] - start) // BIN_SECONDS * BIN_SECONDS / 60)
    real = frame[frame["name"] != SCHEDULE_LAG]
    lag = frame[frame["name"] == SCHEDULE_LAG]
    fig, (latency, throughput, offered, behind) = time_axes(4, "Client side", duration)

    latency.set_title("request latency")
    grouped = real.groupby("bin")["latency_ms"]
    plot_lines(latency, pd.DataFrame({f"p{q}": grouped.quantile(q / 100) for q in (50, 95, 99)}))
    latency.set_ylabel("ms")

    throughput.set_title("completed requests by outcome")
    status = real["status"]
    polling = (real["name"] == SECURE_POLL) & status.isin([404, 409])
    classes = np.select([((status >= 200) & (status < 300)) | polling, status == 429], ["ok", "429"], "error")
    rates = real.assign(cls=classes).groupby(["bin", "cls"]).size().unstack("cls", fill_value=0) / BIN_SECONDS
    plot_lines(throughput, rates)
    throughput.set_ylabel("req/s")

    offered.set_title("iterations started")
    plot_lines(offered, (lag.groupby("bin").size() / BIN_SECONDS).to_frame("iterations"))
    offered.set_ylabel("iter/s")

    behind.set_title("schedule lag")
    lag_seconds = lag.groupby("bin")["latency_ms"]
    plot_lines(behind, pd.DataFrame({"median": lag_seconds.median(), "max": lag_seconds.max()}) / 1000)
    behind.set_ylabel("s")
    save(fig, path)


def plot_containers(prom: dict[str, pd.DataFrame], start: float, duration: float, path: Path) -> None:
    fig, (cpu, memory) = time_axes(2, "Containers", duration, height=3)
    colors = SERVICE_COLORS | {"monitoring": MUTED}
    for ax, name, scale, unit in ((cpu, "container_cpu", 1, "cores"), (memory, "container_memory", 2**20, "MiB")):
        table = wide(prom[name], start, [SERVICE]) / scale
        if not table.empty:
            others = [column for column in table.columns if column not in SERVICES]
            table = table.drop(columns=others).assign(monitoring=table[others].sum(axis=1, min_count=1))
        ax.set_title(f"{name.split('_')[1]} per service")
        plot_lines(ax, table, colors)
        ax.set_ylabel(unit)
    save(fig, path)


def plot_gateways(prom: dict[str, pd.DataFrame], start: float, duration: float, path: Path) -> None:
    fig, (rate, share) = time_axes(2, "Gateways", duration)
    table = wide(prom["requests"], start, ["instance"])
    rate.set_title("requests per gateway")
    plot_lines(rate, table, SERVICE_COLORS)
    rate.set_ylabel("req/s")
    share.set_title("share of requests")
    plot_lines(share, table.div(table.sum(axis=1), axis=0) * 100 if not table.empty else table, SERVICE_COLORS)
    share.set_ylabel("%")
    save(fig, path)


def plot_queues(prom: dict[str, pd.DataFrame], start: float, duration: float, path: Path) -> None:
    fig, (depth, service, fanout) = time_axes(3, "Task queues", duration)
    depth.set_title("broker queue depth")
    plot_lines(depth, wide(prom["queue_depth"], start, ["key"]))
    depth.set_ylabel("tasks")

    tasks = prom["task_rate"]
    service.set_title("tasks completed")
    plot_lines(service, wide(tasks, start, ["name"]))
    service.set_ylabel("tasks/s")

    fanout.set_title("aggregation tasks completed per worker")
    aggregations = tasks[tasks["name"].str.rsplit(".", n=1).str[-1] == AGGREGATION_TASK] if not tasks.empty else tasks
    plot_lines(fanout, wide(aggregations, start, ["hostname"]), SERVICE_COLORS)
    fanout.set_ylabel("tasks/s")
    save(fig, path)


def plot_stages(runs: pd.DataFrame, cohort: str, last: str, outcome: str, title: str, path: Path) -> None:
    runs = runs[runs["outcome"] == outcome]
    if runs.empty:
        return
    stages = list(runs.columns[runs.columns.get_loc(last) + 1:])
    totals = runs[stages].sum().sort_values(ascending=False)
    shown = [stage for stage in stages if stage in totals.index[:len(PALETTE) - 1]]
    runs = runs.assign(**{
        "other stages": runs[[s for s in stages if s not in shown]].sum(axis=1),
        "unaccounted": (runs["total_seconds"] - runs[stages].sum(axis=1)).clip(lower=0),
    })
    segments = [(name, color) for name, color in
                [*zip(shown, PALETTE), ("other stages", PALETTE[len(shown)]), ("unaccounted", MUTED)]
                if runs[name].sum() > 0]
    models = sorted(runs["model_key"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 2.4 * len(models) + 0.6), layout="constrained", squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    for ax, model in zip(axes[:, 0], models):
        part = runs[runs["model_key"] == model].sort_values([cohort, "started_at"])
        x = np.arange(len(part))
        bottom = np.zeros(len(part))
        for name, color in segments:
            values = part[name].fillna(0).to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, width=0.7, color=color, edgecolor=SURFACE, linewidth=1, label=name)
            bottom += values
        every = max(1, len(part) // 20)
        ax.set_xticks(x[::every], part[cohort].astype(int).to_numpy()[::every])
        ax.grid(axis="x", visible=False)
        ax.set_title(model)
        ax.set_ylabel("seconds")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[-1, 0].set_xlabel(f"{cohort} (one bar per task)")
    save(fig, path)


def plot_429(prom: dict[str, pd.DataFrame], workers: dict[str, pd.DataFrame], start: pd.Timestamp,
             duration: float, path: Path) -> None:
    fig, (ax,) = time_axes(1, "Rate limiting", duration, height=3)
    ax.set_title("429 share of requests per handler")
    plot_lines(ax, wide(prom["ratio_429"], start.timestamp(), ["handler"]) * 100)
    ax.set_ylabel("%")
    frame = workers.get(AGGREGATION_TASK)
    if frame is not None and not frame.empty:
        minutes = (pd.to_datetime(frame["started_at"], utc=True, format="ISO8601") - start).dt.total_seconds() / 60
        for index, minute in enumerate(sorted(set(minutes.round(1)))):
            ax.axvline(minute, color="#c3c2b7", linewidth=0.8, linestyle=":",
                       label=f"{AGGREGATION_TASK} started" if index == 0 else None)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def plot_secure_sessions(sessions: pd.DataFrame, start: pd.Timestamp, duration: float, path: Path) -> None:
    fig, (ax,) = time_axes(1, "Secure sessions", duration, height=3)
    ax.set_title("sessions by outcome, binned by creation time")
    if sessions.empty:
        no_data(ax)
        save(fig, path)
        return
    minutes = (pd.to_datetime(sessions["created_at"], utc=True, format="ISO8601") - start).dt.total_seconds() / 60
    width = max(1.0, np.ceil(duration / 40))
    outcome = sessions["status"].where(sessions["status"].isin(["summed", "failed"]), "in flight")
    counts = (sessions.assign(bin=minutes // width * width, outcome=outcome)
              .groupby(["bin", "outcome"]).size().unstack("outcome", fill_value=0))
    bottom = np.zeros(len(counts))
    for name, color in STATUS_COLORS.items():
        if name in counts:
            ax.bar(counts.index + width / 2, counts[name], bottom=bottom, width=width * 0.85, color=color,
                   edgecolor=SURFACE, linewidth=1, label=name)
            bottom += counts[name].to_numpy()
    ax.set_ylabel("sessions")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def plot_overview(prom: dict[str, pd.DataFrame], start: float, duration: float, path: Path) -> None:
    columns = 2
    rows = -(-len(prom) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 2.3 * rows), sharex=True, layout="constrained")
    for ax, (name, frame) in zip(axes.flat, prom.items()):
        ax.set_title(name)
        mark_stages(ax)
        plot_lines(ax, wide(frame, start))
        ax.set_xlim(0, duration)
    for ax in axes.flat[len(prom):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("minutes since start")
    save(fig, path)


def window(manifest: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    return utc(manifest["start"]), utc(manifest["end"])


def dump(run_dir: Path, prometheus: str) -> None:
    start, end = window(json.loads((run_dir / "manifest.json").read_text()))
    for name, query in QUERIES.items():
        frame = query_range(prometheus.rstrip("/"), query, start.timestamp(), end.timestamp())
        frame.assign(timestamp=pd.to_datetime(frame["timestamp"], unit="s", utc=True)).to_csv(
            run_dir / f"prometheus_{name}.csv", index=False)
        series = len(frame.drop(columns=["timestamp", "value"]).drop_duplicates()) if not frame.empty else 0
        print(f"{name}: {series} series, {len(frame)} samples")
    for task, frame in cut_worker_metrics(run_dir, start, end).items():
        print(f"worker_{task}: {len(frame)} rows")


def load_prometheus(run_dir: Path) -> dict[str, pd.DataFrame]:
    prom = {}
    for name in QUERIES:
        frame = pd.read_csv(run_dir / f"prometheus_{name}.csv")
        stamps = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        prom[name] = frame.assign(timestamp=(stamps - pd.Timestamp(0, tz="UTC")).dt.total_seconds())
    return prom


def load_worker_metrics(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {path.stem.removeprefix("worker_"): pd.read_csv(path) for path in sorted(run_dir.glob("worker_*.csv"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a benchmark run's metrics and plot them")
    parser.add_argument("run_id")
    parser.add_argument("--prometheus", default="http://localhost:9090", help="Prometheus base URL")
    parser.add_argument("--requery", action="store_true",
                        help="query Prometheus again even if the run already holds its series")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    start, end = window(manifest)
    duration = (end - start).total_seconds() / 60

    if args.requery or not all((run_dir / f"prometheus_{name}.csv").exists() for name in QUERIES):
        dump(run_dir, args.prometheus)
    prom = load_prometheus(run_dir)
    workers = load_worker_metrics(run_dir)
    stage_marks[:] = [(utc(stage["start"]) - start).total_seconds() / 60 for stage in manifest.get("stages", [])[1:]]
    if "load_end" in manifest:
        stage_marks.append((utc(manifest["load_end"]) - start).total_seconds() / 60)

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    origin = start.timestamp()
    request_logs = sorted(run_dir.glob(REQUEST_LOGS))
    if request_logs:
        requests = pd.concat([pd.read_csv(log) for log in request_logs], ignore_index=True)
        plot_client(requests, origin, duration, figures / "client.png")
    plot_containers(prom, origin, duration, figures / "containers.png")
    plot_gateways(prom, origin, duration, figures / "gateways.png")
    plot_queues(prom, origin, duration, figures / "queues.png")
    for task, (cohort, last, outcome) in STAGED_TASKS.items():
        if task in workers:
            plot_stages(workers[task], cohort, last, outcome, f"{task} stages", figures / f"stages_{task}.png")
    plot_429(prom, workers, start, duration, figures / "ratio_429.png")
    if (run_dir / SECURE_SESSIONS).exists():
        plot_secure_sessions(pd.read_csv(run_dir / SECURE_SESSIONS), start, duration,
                             figures / "secure_sessions.png")
    plot_overview(prom, origin, duration, figures / "overview.png")


if __name__ == "__main__":
    main()

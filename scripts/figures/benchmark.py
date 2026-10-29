import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RUNS = Path("results/benchmark")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_COLORS = {"covertype-nb": BLUE, "feature-ae": ORANGE, "feature-mlp": AQUA, "mnist-mlp": YELLOW}

plt.switch_backend("Agg")
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titleweight": "bold", "font.size": 12,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 11,
})


def es(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


def es_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    formatter = FuncFormatter(lambda value, _: es(value, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def manifest(run: str) -> dict:
    return json.loads((RUNS / run / "manifest.json").read_text())


def seconds(frame: pd.DataFrame, column: str = "timestamp") -> pd.Series:
    return pd.to_datetime(frame[column], utc=True, format="ISO8601").map(pd.Timestamp.timestamp)


def prometheus(run: str, name: str) -> pd.DataFrame:
    frame = pd.read_csv(RUNS / run / f"prometheus_{name}.csv")
    return frame.assign(t=seconds(frame))


def requests(run: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(path) for path in (RUNS / run).glob("requests_*.csv")], ignore_index=True)
    return frame[frame["name"] != "schedule_lag"]


def host_cpu(cpu: pd.DataFrame, prefix: str, start: float, end: float) -> float:
    part = cpu[cpu["instance"].str.startswith(prefix) & (cpu.t >= start) & (cpu.t < end)]
    return part.groupby("t")["value"].mean().mean() * 100


def stage_table(run: str) -> pd.DataFrame:
    meta = manifest(run)
    period = meta["load"].get("period") or meta["load"]["interval"]
    log = requests(run)
    polled = (log["name"] == "secure/session") & log["status"].isin([404, 409])
    useful = log["status"].between(200, 299) | polled
    cpu = prometheus(run, "host_cpu")
    sessions = None
    if (RUNS / run / "secure_sessions.csv").exists():
        sessions = pd.read_csv(RUNS / run / "secure_sessions.csv")
        sessions = sessions[sessions["status"] == "summed"]
        sessions = sessions.assign(t=seconds(sessions, "finished_at"))
    rows = []
    peak = 0
    for stage in meta["stages"][1:]:
        start = pd.Timestamp(stage["start"]).timestamp() + 20
        end = pd.Timestamp(stage["end"]).timestamp()
        if stage["active"] < max(100, peak):
            continue
        peak = stage["active"]
        part = log[(log["timestamp"] >= start) & (log["timestamp"] < end)]
        rows.append({
            "users": stage["active"],
            "participations": stage["active"] / period,
            "rps": len(part) / (end - start),
            "useful": useful.loc[part.index].sum() / (end - start),
            "p95": part["latency_ms"].quantile(0.95),
            "gateway_cpu": host_cpu(cpu, "fastapi", start, end),
            "worker_cpu": host_cpu(cpu, "celery", start, end),
            "offered": stage["active"] / period * len(meta["models"]),
            "effective": (sessions[(sessions.t >= start) & (sessions.t < end)]["member_count"].sum() / (end - start)
                          if sessions is not None else np.nan),
        })
    return pd.DataFrame(rows)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def label_end(ax, x, y, text: str, color: str, dy: float = 0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                color=INK_2, fontsize=11)
    ax.plot([x], [y], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=2)


def topology_scaling(out: Path) -> None:
    series = [("1×1 (4 vCPU)", "20261011-021128-1x1-light-secure", BLUE),
              ("2×1 (8 vCPU)", "20261011-024910-2x1-light-secure", ORANGE),
              ("4×1 (8 vCPU)", "20261011-031434-4x1-light-secure", AQUA)]
    fig, (rate, total, latency) = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for label, run, color in series:
        table = stage_table(run)
        rate.plot(table.offered, table.useful, "-o", color=color, label=label, markeredgecolor=SURFACE)
        total.plot(table.offered, table.rps, "-o", color=color, label=label, markeredgecolor=SURFACE)
        latency.plot(table.offered, table.p95, "-o", color=color, label=label, markeredgecolor=SURFACE)
    rate.set_title("Throughput útil")
    rate.set_ylabel("solicitudes/s (sin 429)")
    total.set_title("Throughput total")
    total.set_ylabel("solicitudes/s (con 429 y errores)")
    latency.set_title("Latencia p95 del cliente")
    latency.set_yscale("log")
    latency.set_ylabel("ms (escala log.)")
    latency.yaxis.set_major_formatter(FuncFormatter(lambda value, _: es(value)))
    for ax in (rate, total, latency):
        ax.set_xlabel("participaciones por segundo")
        es_axis(ax, "x")
    es_axis(rate)
    es_axis(total)
    rate.legend(loc="upper left")
    save(fig, out, "escalado-topologias")


def export_memory(out: Path) -> None:
    series = [("antes de la corrección", "20261010-201951-1x1-light-quantize", ORANGE),
              ("después de la corrección", "20261011-023242-1x1-light-quantize", BLUE)]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for label, run, color in series:
        exports = []
        for task, outcome in (("quantize_submission", "done"), ("federated_aggregation", "aggregated")):
            frame = pd.read_csv(RUNS / run / f"worker_{task}.csv")
            frame = frame[frame["outcome"] == outcome]
            exports.append(seconds(frame, "started_at") + frame["total_seconds"])
        done = np.sort(pd.concat(exports).to_numpy())
        memory = prometheus(run, "container_memory")
        service = [column for column in memory.columns if "service" in column][0]
        memory = memory[memory[service] == "celery-1"].sort_values("t")
        count = np.searchsorted(done, memory.t.to_numpy())
        ax.plot(count, memory["value"] / 2**30, color=color, label=label)
        label_end(ax, count[-1], memory["value"].iloc[-1] / 2**30,
                  f"{es(memory['value'].iloc[-1] / 2**30, 1)} GiB", color)
    ax.set_title("Memoria del worker según los trabajos con exportación TFLite")
    ax.set_xlabel("trabajos acumulados que exportan TFLite (cuantizaciones y rondas)")
    ax.set_ylabel("memoria del worker (GiB)")
    ax.set_ylim(bottom=0)
    es_axis(ax, "x")
    es_axis(ax, decimals=0)
    ax.legend(loc="center right")
    save(fig, out, "fuga-memoria-worker")


def quantize_vs_secure(out: Path) -> None:
    series = [("quantize", "20261011-023242-1x1-light-quantize"),
              ("secure", "20261011-021128-1x1-light-secure")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained", sharey=True)
    for ax, (title, run) in zip(axes, series):
        meta = manifest(run)
        rate = len(meta["models"]) / (meta["load"].get("period") or meta["load"]["interval"])
        cpu = prometheus(run, "host_cpu")
        rows = []
        for stage in meta["stages"][1:-1]:
            start = pd.Timestamp(stage["start"]).timestamp() + 20
            end = pd.Timestamp(stage["end"]).timestamp()
            rows.append((stage["active"] * rate, host_cpu(cpu, "celery", start, end), host_cpu(cpu, "fastapi", start, end)))
        table = pd.DataFrame(rows, columns=["users", "worker", "gateway"]).groupby("users").max().reset_index()
        ax.plot(table.users, table.worker, "-o", color=ORANGE, label="CPU del worker", markeredgecolor=SURFACE)
        ax.plot(table.users, table.gateway, "-o", color=BLUE, label="CPU del gateway", markeredgecolor=SURFACE)
        ax.set_title(title)
        ax.set_xticks(table.users)
        ax.set_xlabel("participaciones por segundo")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: es(value, 0 if value >= 10 else 1)))
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("uso de CPU (%)")
    axes[0].legend(loc="upper left")
    save(fig, out, "quantize-vs-secure")


def session_outcomes(out: Path) -> None:
    def counts(run: str) -> tuple[int, int]:
        detail = json.loads((RUNS / run / "checks.json").read_text())["secure_sessions"]["detail"]
        summed, total = map(int, re.match(r"(\d+)/(\d+) summed", detail).groups())
        return summed, total - summed

    rows = [("campaña 1: ronda por temporizador cada 30 s\n(sin fallas forzadas)", "20261011-054552-4x1-light-secure-longfail"),
            ("final: ronda cerrada por conteo\n(10 % de fallas forzadas)", "20261011-a-4x1-light-secure-count")]
    fig, ax = plt.subplots(figsize=(11, 3.4), layout="constrained")
    for index, (label, run) in enumerate(rows):
        summed, failed = counts(run)
        share = summed / (summed + failed)
        ax.barh(index, share * 100, color=BLUE, height=0.55, label="sumadas" if index == 0 else None)
        ax.barh(index, (1 - share) * 100, left=share * 100, color=ORANGE, height=0.55,
                label="fallidas" if index == 0 else None, edgecolor=SURFACE, linewidth=2)
        ax.text(share * 50, index, f"{es(share * 100)} %", ha="center", va="center", color="white",
                fontweight="bold")
        ax.text(share * 100 + (1 - share) * 50, index, f"{es((1 - share) * 100)} %", ha="center",
                va="center", color="white", fontweight="bold")
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("sesiones seguras (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Sesiones de agregación segura que terminan sumadas")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    save(fig, out, "sesiones-seguras")


def limit_regression(out: Path) -> None:
    series = [("consulta sin LIMIT", "buggy-20261011-a", ORANGE), ("con LIMIT 1", "20261011-a-4x1-light-secure-count", BLUE)]
    fig, (cpu_ax, sql_ax, latency_ax) = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True, layout="constrained")
    for label, run, color in series:
        start = pd.Timestamp(manifest(run)["start"]).timestamp()
        end = pd.Timestamp(manifest(run)["load_end"]).timestamp()
        cpu = prometheus(run, "host_cpu")
        cpu = cpu[cpu["instance"].str.startswith("fastapi") & (cpu.t <= end)].groupby("t")["value"].mean()
        cpu = cpu.rolling(30, min_periods=1).mean()
        cpu_ax.plot((cpu.index - start) / 60, cpu * 100, color=color, label=label)
        sql = pd.read_csv(RUNS / run / "gcp_sql_cpu.csv")
        sql = sql.assign(t=seconds(sql))
        sql = sql[(sql.t >= start) & (sql.t <= end)]
        sql_ax.plot((sql.t - start) / 60, sql["value"] * 100, color=color, label=label)
        log = requests(run)
        log = log[log["timestamp"] <= end]
        bins = ((log["timestamp"] - start) // 30 * 30 / 60)
        p95 = log.groupby(bins)["latency_ms"].quantile(0.95)
        latency_ax.plot(p95.index, p95 / 1000, color=color, label=label)
    cpu_ax.set_title("CPU media de los gateways")
    cpu_ax.set_ylabel("uso de CPU (%)")
    cpu_ax.set_ylim(0, 100)
    sql_ax.set_title("CPU de Cloud SQL")
    sql_ax.set_ylabel("uso de CPU (%)")
    sql_ax.set_ylim(0, 100)
    latency_ax.set_title("Latencia p95 del cliente")
    latency_ax.set_ylabel("segundos")
    latency_ax.set_xlabel("minutos desde el inicio (misma carga constante: 4 000 usuarios)")
    latency_ax.set_xlim(left=4)
    latency_ax.set_ylim(0, 13)
    cpu_ax.legend(loc="upper left")
    save(fig, out, "costo-por-ronda")


def final_staircase(out: Path) -> None:
    series = [("campaña 1: 4×1, rondas de 30 s", "20261011-031434-4x1-light-secure", MUTED),
              ("campaña 1: 4×1, plazos largos", "20261011-054552-4x1-light-secure-longfail", ORANGE),
              ("final: 4×1, cierre por conteo", "20261011-c-4x1-light-secure-staircase", BLUE)]
    fig, (rate, effective, latency) = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    top = 0
    for label, run, color in series:
        table = stage_table(run)
        top = max(top, table.offered.max())
        for ax, column in ((rate, "rps"), (effective, "effective"), (latency, "p95")):
            ax.plot(table.offered, table[column], "-o", color=color, label=label, markeredgecolor=SURFACE)
    effective.plot([0, top], [0, top], linestyle="--", color="#c3c2b7", linewidth=1, zorder=0)
    effective.text(62, 70, "todo lo ofrecido llega", color=MUTED, rotation=38, fontsize=10)
    knee = stage_table("20261011-c-4x1-light-secure-staircase").query("users == 4000").iloc[0]
    rate.annotate(f"{es(knee.rps)} sol./s", (knee.offered, knee.rps), xytext=(20, -50),
                  textcoords="offset points", color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED})
    effective.annotate(f"{es(knee.effective)} aportes/s\n({es(100 * knee.effective / knee.offered)} % de lo ofrecido)",
                       (knee.offered, knee.effective), xytext=(25, -60), textcoords="offset points",
                       color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED})
    latency.annotate(f"p95 {es(knee.p95)} ms", (knee.offered, knee.p95), xytext=(20, -40),
                     textcoords="offset points", color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED})
    rate.set_title("Solicitudes atendidas")
    rate.set_ylabel("solicitudes por segundo")
    rate.set_ylim(bottom=0)
    es_axis(rate)
    effective.set_title("Aportes que llegan a una ronda")
    effective.set_ylabel("aportes agregados por segundo")
    effective.set_ylim(0, top * 0.85)
    latency.set_title("Latencia p95 del cliente")
    latency.set_yscale("log")
    latency.set_ylabel("ms (escala log.)")
    latency.yaxis.set_major_formatter(FuncFormatter(lambda value, _: es(value)))
    for ax in (rate, effective, latency):
        ax.set_xlabel("aportes ofrecidos por segundo")
    rate.legend(loc="lower right")
    fig.suptitle("Mismo hardware (8 vCPU de gateways): más solicitudes atendidas y aportes que ya no colapsan bajo carga",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "escalera-final")


CONTRIBUTION_STEPS = [("weights", "descargar pesos"), ("download/{artifact}", "descargar 2 artefactos"),
                      ("secure/join", "unirse a la sesión"), ("secure/session", "consultar la sesión"),
                      ("secure/submit", "enviar el vector")]


def contribution_cost(out: Path) -> None:
    run = "20261011-c-4x1-light-secure-staircase"
    meta = manifest(run)
    stage = next(stage for stage in meta["stages"] if stage["active"] == 4000)
    start = pd.Timestamp(stage["start"]).timestamp() + 20
    end = pd.Timestamp(stage["end"]).timestamp()
    log = requests(run)
    log = log[(log["timestamp"] >= start) & (log["timestamp"] < end)]
    offered = stage["active"] / meta["load"]["period"] * len(meta["models"]) * (end - start)
    counts = log.groupby("name").size() / offered
    downloaded = log.groupby("name")["response_bytes"].sum() / offered / 1024
    uploaded = sum(meta["models"].values()) * 4 / len(meta["models"]) / 1024
    colors = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4"]
    fig, (requests_ax, bytes_ax) = plt.subplots(2, 1, figsize=(12, 3.2), layout="constrained")
    left = 0.0
    for (name, label), color in zip(CONTRIBUTION_STEPS, colors):
        value = counts.get(name, 0)
        requests_ax.barh(0, value, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
        requests_ax.text(left + value / 2, 0, f"{label}\n{es(value, 1)}", ha="center", va="center", fontsize=10,
                         color="white" if color in (BLUE, ORANGE) else INK)
        left += value
    requests_ax.set_xlim(0, left)
    requests_ax.set_yticks([])
    requests_ax.set_title(f"Solicitudes por aporte: {es(left, 1)} en promedio")
    requests_ax.grid(False)
    left = 0.0
    for value, label, color in ((downloaded.sum(), "descargado (pesos, artefactos y sesión)", BLUE),
                                (uploaded, "subido (vector enmascarado)", ORANGE)):
        bytes_ax.barh(0, value, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
        if label:
            bytes_ax.text(left + value / 2, 0, f"{label}\n{es(value)} KB", ha="center", va="center", fontsize=10,
                          color="white")
        left += value
    bytes_ax.set_xlim(0, left)
    bytes_ax.set_yticks([])
    bytes_ax.set_title(f"Datos por aporte: {es(left)} KB en promedio (mnist-mlp pesa casi todo)")
    bytes_ax.grid(False)
    for ax in (requests_ax, bytes_ax):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Qué es un aporte: un usuario, un modelo, una ronda (agregación segura)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "costo-de-un-aporte")


def count_closed_rounds(out: Path) -> None:
    run = "20261011-a-4x1-light-secure-count"
    start = pd.Timestamp(manifest(run)["start"]).timestamp()
    load_end = pd.Timestamp(manifest(run)["load_end"]).timestamp()
    fig, (rounds, size) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, layout="constrained")
    agg = pd.read_csv(RUNS / run / "worker_federated_aggregation.csv")
    agg = agg[agg["outcome"] == "aggregated"].assign(t=lambda frame: seconds(frame, "started_at"))
    for key, color in MODEL_COLORS.items():
        part = agg[agg["model_key"] == f"{key}-secure"].sort_values("t")
        minutes = (part.t - start) / 60
        rounds.step(minutes, np.arange(1, len(part) + 1), where="post", color=color, label=key)
    rounds.set_title("Rondas agregadas por modelo (límite de 1 000 participantes)")
    rounds.set_ylabel("rondas acumuladas")
    rounds.legend(loc="upper left", ncols=2)
    db = prometheus(run, "pg_database_size")
    size.plot((db.t - start) / 60, db["value"] / 2**30, color=BLUE)
    size.set_title("Tamaño de la base de datos (los aportes se borran al cerrar cada ronda)")
    size.set_ylabel("GiB")
    size.set_ylim(0, 1.5)
    es_axis(size, decimals=1)
    size.set_xlabel("minutos desde el inicio (4 000 usuarios, cada uno participa cada 5 min)")
    for ax in (rounds, size):
        ax.axvline((load_end - start) / 60, color="#c3c2b7", linestyle="--", linewidth=1)
    fig.suptitle("Rondas que cierran solas a ritmo constante, con la base de datos plana",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "rondas-por-conteo")


def pool_default(out: Path) -> None:
    series = [("pool por defecto (5 + 10)", "20261011-005647-1x1-light-secure", ORANGE),
              ("pool ajustado (40 + 10)", "20261011-021128-1x1-light-secure", BLUE)]
    fig, (rate, cpu) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for label, run, color in series:
        table = stage_table(run)
        rate.plot(table.users, table.useful, "-o", color=color, label=label, markeredgecolor=SURFACE)
        cpu.plot(table.users, table.gateway_cpu, "-o", color=color, label=label, markeredgecolor=SURFACE)
    rate.set_title("Throughput útil")
    rate.set_ylabel("solicitudes/s (sin 429)")
    cpu.set_title("CPU del gateway")
    cpu.set_ylabel("uso de CPU (%)")
    cpu.set_ylim(0, 105)
    for ax in (rate, cpu):
        ax.set_xlabel("usuarios simulados activos")
        es_axis(ax, "x")
    rate.legend(loc="lower right")
    fig.suptitle("Topología 1×1: con el pool por defecto el throughput se estanca con la CPU al 60 %",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "anexo-pool-conexiones")


def pool_deadlock(out: Path) -> None:
    run = "aborted-20261011-a"
    start = pd.Timestamp(manifest(run)["start"]).timestamp()
    fig, (rate, conns) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, layout="constrained")
    traffic = prometheus(run, "requests")
    traffic = traffic.assign(kind=np.where(traffic["status"].astype(str).str.startswith("5"), "error 500", "resto"))
    table = traffic.groupby(["t", "kind"])["value"].sum().unstack("kind").rolling(15, min_periods=1).mean()
    rate.plot((table.index - start) / 60, table["resto"], color=BLUE, label="respuestas sin error 5xx")
    rate.plot((table.index - start) / 60, table["error 500"] * 10, color=ORANGE, label="errores 500 (×10)")
    rate.set_title("Solicitudes por segundo")
    rate.set_ylabel("solicitudes/s")
    rate.legend(loc="center right")
    pg = prometheus(run, "pg_connections")
    pg = pg[pg["state"] == "idle in transaction"].groupby("t")["value"].sum()
    conns.plot((pg.index - start) / 60, pg, color=ORANGE)
    conns.set_title("Conexiones a Postgres «idle in transaction» (sesiones retenidas)")
    conns.set_ylabel("conexiones")
    conns.set_xlabel("minutos desde el inicio (ráfaga de 4 000 inicios de sesión a 50/s)")
    save(fig, out, "anexo-interbloqueo")


def aggregation_timing(runs: dict[int, str], model: str) -> pd.DataFrame:
    rows = []
    for cohort, run in runs.items():
        frame = pd.read_csv(RUNS / run / "worker_federated_aggregation.csv")
        frame = frame[(frame["outcome"] == "aggregated") & (frame["model_key"] == model) & (frame["cohort"] >= cohort * 0.99)]
        memory = prometheus(run, "container_memory")
        service = [column for column in memory.columns if "service" in column][0]
        rows.append({"cohort": cohort, "blocks": frame["blocks"].median(), "read": frame["blob_fetch"].median(),
                     "trim": frame["trimmed_mean"].median(), "total": frame["total_seconds"].median(),
                     "memory": memory[memory[service] == "celery-1"]["value"].max() / 2**30})
    return pd.DataFrame(rows)


FIXED = {500: "20261011-b1-4x1-light-raw-l500", 1000: "20261011-b2-4x1-light-raw-l1000",
         2000: "20261012-b3f-4x1-light-raw-l2000", 4000: "20261012-b4f-4x1-light-raw-l4000"}
UNFIXED = {500: "20261011-b1-4x1-light-raw-l500", 1000: "20261011-b2-4x1-light-raw-l1000",
           2000: "20261011-b3-4x1-light-raw-l2000", 4000: "20261011-b4-4x1-light-raw-l4000"}


def block_sort(out: Path) -> None:
    before = aggregation_timing(UNFIXED, "mnist-mlp")
    after = aggregation_timing(FIXED, "mnist-mlp")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(len(before))
    ax.bar(x - 0.2, before["trim"], width=0.38, color=ORANGE, label="bloque en orden C (por filas)")
    ax.bar(x + 0.2, after["trim"], width=0.38, color=BLUE, label="bloque en orden Fortran (por columnas)")
    for index, (old, new) in enumerate(zip(before["trim"], after["trim"])):
        ax.text(index - 0.2, old, f"{es(old, 2)} s", ha="center", va="bottom", color=INK_2, fontsize=10)
        ax.text(index + 0.2, new, f"{es(new, 2)} s", ha="center", va="bottom", color=INK_2, fontsize=10)
    ax.set_xticks(x, [f"{es(c)} filas\n{int(b)} bloque{'s' if b > 1 else ''}" for c, b in zip(after["cohort"], after["blocks"])])
    ax.set_ylabel("media recortada por ronda (s)")
    ax.grid(axis="x", visible=False)
    ax.set_title("mnist-mlp: tiempo de la media recortada según el tamaño de la ronda")
    ax.legend(loc="upper left")
    save(fig, out, "anexo-orden-de-bloques")


def cohort_scaling(out: Path) -> None:
    table = aggregation_timing(FIXED, "mnist-mlp")
    fig, (time_ax, memory_ax) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    time_ax.plot(table.cohort, table.read, "-o", color=BLUE, label="lectura por bloques", markeredgecolor=SURFACE)
    time_ax.plot(table.cohort, table.total, "-o", color=ORANGE, label="ronda completa", markeredgecolor=SURFACE)
    time_ax.plot(table.cohort, table.trim, "-o", color=AQUA, label="media recortada", markeredgecolor=SURFACE)
    time_ax.set_title("Duración de la ronda de mnist-mlp")
    time_ax.set_ylabel("segundos (mediana por ronda)")
    time_ax.legend(loc="upper left")
    memory_ax.bar(range(len(table)), table.memory, color=BLUE, width=0.55)
    for index, value in enumerate(table.memory):
        memory_ax.text(index, value, f"{es(value, 1)} GiB", ha="center", va="bottom", color=INK_2)
    memory_ax.set_xticks(range(len(table)), [f"{es(c)}\n({int(b)} bl.)" for c, b in zip(table.cohort, table.blocks)])
    memory_ax.set_title("Memoria máxima del worker")
    memory_ax.set_ylabel("GiB")
    memory_ax.set_ylim(0, 4)
    es_axis(memory_ax, decimals=1)
    memory_ax.grid(axis="x", visible=False)
    for ax in (time_ax, memory_ax):
        ax.set_xlabel("aportes por ronda")
    es_axis(time_ax, "x")
    fig.suptitle("Rondas grandes: el tiempo crece lineal con la cohorte y la memoria queda acotada",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, out, "anexo-cohortes-grandes")


def managed_services(out: Path) -> None:
    runs = [("A", "20261011-a-4x1-light-secure-count"), ("B 500", FIXED[500]), ("B 1 000", FIXED[1000]),
            ("B 2 000", FIXED[2000]), ("B 4 000", FIXED[4000]), ("C", "20261011-c-4x1-light-secure-staircase")]
    sql = [pd.read_csv(RUNS / run / "gcp_sql_cpu.csv")["value"].max() * 100 for _, run in runs]
    redis = [pd.read_csv(RUNS / run / "gcp_redis_cpu.csv")["value"].max() * 100 for _, run in runs]
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    x = np.arange(len(runs))
    ax.bar(x - 0.2, sql, width=0.38, color=BLUE, label="Cloud SQL (Postgres, 8 vCPU)")
    ax.bar(x + 0.2, redis, width=0.38, color=ORANGE, label="Memorystore (Valkey)")
    for index, value in enumerate(sql):
        ax.text(index - 0.2, value, f"{es(value)} %", ha="center", va="bottom", color=INK_2, fontsize=10)
    ax.set_xticks(x, [label for label, _ in runs])
    ax.set_ylabel("CPU máxima (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title("Servicios administrados: CPU máxima por corrida final")
    ax.legend(loc="upper left")
    save(fig, out, "anexo-servicios-administrados")


FIGURES = {
    "escalado-topologias": topology_scaling,
    "fuga-memoria-worker": export_memory,
    "quantize-vs-secure": quantize_vs_secure,
    "sesiones-seguras": session_outcomes,
    "costo-por-ronda": limit_regression,
    "escalera-final": final_staircase,
    "costo-de-un-aporte": contribution_cost,
    "rondas-por-conteo": count_closed_rounds,
    "anexo-pool-conexiones": pool_default,
    "anexo-interbloqueo": pool_deadlock,
    "anexo-orden-de-bloques": block_sort,
    "anexo-cohortes-grandes": cohort_scaling,
    "anexo-servicios-administrados": managed_services,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the benchmark figures for the thesis defense")
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", default="", help="comma-separated figure names")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        if not args.only or name in args.only.split(","):
            plot(args.out)


if __name__ == "__main__":
    main()

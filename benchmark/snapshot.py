import argparse
import json
import subprocess
import urllib.request

from common.config import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot the Prometheus TSDB and copy it into the results")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--container", default="backend_prometheus_1")
    args = parser.parse_args()

    request = urllib.request.Request(f"{args.prometheus.rstrip('/')}/api/v1/admin/tsdb/snapshot", method="POST")
    with urllib.request.urlopen(request, timeout=300) as resp:
        name = json.load(resp)["data"]["name"]
    destination = RESULTS_DIR / "benchmark" / "tsdb"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["podman", "cp", f"{args.container}:/prometheus/snapshots/{name}", str(destination / name)],
                   check=True)
    print(f"snapshot in {destination / name}, serve it with --storage.tsdb.path pointed at it")


if __name__ == "__main__":
    main()

"""Queue a federated aggregation round on the Celery worker, to test or run a
round by hand outside the daily beat schedule:

    uv run -m scripts.queue_aggregation           # every initialized model
    uv run -m scripts.queue_aggregation cnn-ae    # a single model
    uv run -m scripts.queue_aggregation --timeout 600

Waits for the task to finish (via the Celery result backend) and prints the
per-model summary it returns.
"""

import argparse

from worker.celery_app import app

AGGREGATION_TASK = "worker.tasks.federated_aggregation"
DEFAULT_TIMEOUT = 300.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_key", nargs="?", default=None,
                        help="model to aggregate (default: every initialized model)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"seconds to wait for the round (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    result = app.send_task(AGGREGATION_TASK, args=[args.model_key])
    print(f"queued federated aggregation for {args.model_key or 'all models'} "
          f"({result.id}), waiting up to {args.timeout:g}s...")
    summary = result.get(timeout=args.timeout)
    for key, message in summary.items():
        print(f"{key}: {message}")


if __name__ == "__main__":
    main()

"""Queue a federated aggregation round on the Celery worker, to test or run a
round by hand outside the daily beat schedule:

    uv run -m scripts.queue_aggregation           # every initialized model
    uv run -m scripts.queue_aggregation cnn-ae    # a single model
"""

import argparse

from worker.celery_app import app

AGGREGATION_TASK = "worker.tasks.federated_aggregation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_key", nargs="?", default=None,
                        help="model to aggregate (default: every initialized model)")
    args = parser.parse_args()

    app.send_task(AGGREGATION_TASK, args=[args.model_key])
    print(f"queued federated aggregation for {args.model_key or 'all models'}")


if __name__ == "__main__":
    main()

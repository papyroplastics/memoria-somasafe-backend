"""Queue a federated aggregation round by hand, for testing — the same
``federated_aggregation`` task the daily beat runs, but on demand. Blocks on the task and
prints its per-model summary. Requires the worker (and its broker/DB) to be up.

    uv run -m scripts.integration.queue_aggregation           # every initialized model
    uv run -m scripts.integration.queue_aggregation cnn-ae    # a single model
"""

import argparse

from common.celery_tasks import FED_AGG_TASK
from ml.model_list import MODELS
from worker.celery_app import app

from scripts.common.api import wait_for_aggregation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?', default=None, choices=sorted(MODELS),
                        help='Model to aggregate (default: every initialized model)')
    args = parser.parse_args()

    result = app.send_task(FED_AGG_TASK, args=[args.model])
    if args.model is not None:
        print(f"{args.model}: {wait_for_aggregation(result, args.model)}")
    else:
        for key, summary in result.get(timeout=300.0).items():
            print(f"{key}: {summary}")


if __name__ == "__main__":
    main()

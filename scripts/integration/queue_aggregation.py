"""Queue a federated aggregation round by hand, for testing. With a model, runs that
model's round and prints its result; without one, runs the dispatcher the beat schedule
fires and prints the models it queued. Requires the worker (and its broker/DB) to be up."""

import argparse

from common.celery_tasks import FED_DISPATCH_TASK
from ml.model_list import MODELS
from worker.celery_app import app

from scripts.common.api import wait_for_aggregation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?', default=None, choices=sorted(MODELS),
                        help='Model to aggregate (default: dispatch every dense model)')
    args = parser.parse_args()

    if args.model is not None:
        result = wait_for_aggregation(app, args.model)
        print(f"{args.model}: {result['outcome']} ({result['detail']})")
    else:
        keys = app.send_task(FED_DISPATCH_TASK).get(timeout=60.0)
        print(f"dispatched: {', '.join(keys) or 'nothing'}")


if __name__ == "__main__":
    main()

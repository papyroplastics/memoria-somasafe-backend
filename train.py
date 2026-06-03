import argparse
from pathlib import Path

from worker.models import mlp, cond_lstm_autoencoder, feature_mlp

# Each model module owns its data prep + training loop and exposes run(result_dir).
RUNNERS = {
    'sine': mlp.run,
    'lstm': cond_lstm_autoencoder.run,
    'feature-mlp': feature_mlp.run,
}


def main():
    parser = argparse.ArgumentParser(
        description='Train a SomaSafe model and export SavedModel + TFLite artifacts '
                    'into results/<model>.')
    parser.add_argument('model', choices=sorted(RUNNERS), help='Model to train')
    args = parser.parse_args()

    result_dir = Path('results') / args.model
    result_dir.mkdir(parents=True, exist_ok=True)
    RUNNERS[args.model](result_dir)


if __name__ == '__main__':
    main()

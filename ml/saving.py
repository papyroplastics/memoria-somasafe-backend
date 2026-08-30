import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from .models.common import TrainableModel, Trainer

def optimize_saved_model(rep_dataset: tf.data.Dataset, saved_dir: Path) -> bytes:
    def rep_iter():
        for feed in rep_dataset:
            yield ('eval', feed)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))  # type: ignore
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # type: ignore
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = rep_iter

    return converter.convert()


def get_trainable_model(model: TrainableModel) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'eval': model.eval.get_concrete_function(),
            'train': model.train.get_concrete_function(),
            'save': model.save.get_concrete_function(),
            'restore': model.restore.get_concrete_function(),
        })

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))  # type: ignore
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  # type: ignore
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # type: ignore
        converter.experimental_enable_resource_variables = True
        return converter.convert()


def get_optimized_model(model: TrainableModel, rep_dataset: tf.data.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'eval': model.eval.get_concrete_function(),
        })

        return optimize_saved_model(rep_dataset, saved_dir)


def save_weights(model: TrainableModel, path: Path) -> None:
    np.save(path, np.asarray(model.save()['weights'], dtype=np.float32))


def load_weights(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def trainable_path(model_dir: Path, tag: str | None = None) -> Path:
    return model_dir / (f'trainable_{tag}.tflite' if tag else 'trainable.tflite')


def weights_path(model_dir: Path, tag: str | None = None) -> Path:
    return model_dir / (f'weights_{tag}.npy' if tag else 'weights.npy')


def save_artifacts(trainer: Trainer, result_dir: Path, postfix: str = '',
                   tflite: bool = True):
    weights_file = result_dir / f'weights{postfix}.npy'
    save_weights(trainer.model, weights_file)
    print(f"Saved weights to {weights_file}")

    if not tflite:
        return

    trainable_file = result_dir / f'trainable{postfix}.tflite'
    trainable_file.write_bytes(get_trainable_model(trainer.model))
    print(f"Saved trainable model to {trainable_file}")

    rep_dataset = trainer.representative_dataset()
    try:
        quantized_file = result_dir / f'quantized{postfix}.tflite'
        quantized_file.write_bytes(get_optimized_model(trainer.model, rep_dataset))
        print(f"Saved quantized model to {quantized_file}")
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")

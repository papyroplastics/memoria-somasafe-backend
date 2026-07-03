import json
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from common.config import NORM_PARAMS_FILE
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


def save_tainable_model(output_dir: Path, model: TrainableModel,
                        postfix: str = '') -> tuple[TrainableModel, Path]:
    saved_model_dir = output_dir / f'trainable-model{postfix}'
    compiled_model_file = output_dir / f'trainable{postfix}.tflite'

    tf.saved_model.save(model, str(saved_model_dir), signatures={
        'eval': model.eval.get_concrete_function(),
        'train': model.train.get_concrete_function(),
        'save': model.save.get_concrete_function(),
        'restore': model.restore.get_concrete_function(),
    })

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))  # type: ignore
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  # type: ignore
    converter.experimental_enable_resource_variables = True
    compiled_model_buf = converter.convert()
    compiled_model_file.write_bytes(compiled_model_buf)

    return tf.saved_model.load(str(saved_model_dir)), saved_model_dir


def save_optimized_model(output_dir: Path, model: TrainableModel,
                         rep_dataset: tf.data.Dataset, postfix: str = ''):
    saved_model_dir = output_dir / f'quantized-model{postfix}'
    compiled_model_file = output_dir / f'quantized{postfix}.tflite'

    tf.saved_model.save(model, str(saved_model_dir), signatures={
        'eval': model.eval.get_concrete_function(),
    })

    compiled_model_file.write_bytes(optimize_saved_model(rep_dataset, saved_model_dir))

    return tf.saved_model.load(str(saved_model_dir))

def load_trainable_weights(tflite_path: Path) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    save = interpreter.get_signature_runner('save')
    return np.asarray(save()['parameters'], dtype=np.float32)


def save_norm_params(result_dir: Path, trainer: Trainer, data_root: Path):
    """Serialize the trainer's model-specific normalization params to norm.json, so the
    gateway can serve them per model over /model/norm and the device applies them at load."""
    path = result_dir / NORM_PARAMS_FILE
    path.write_text(json.dumps(trainer.norm_params(data_root)))
    print(f"saved norm params to {path}")


def save_artifacts(trainer: Trainer, result_dir: Path, eval_dataset: tf.data.Dataset | None,
                   data_root: Path, postfix: str = ''):
    saved_model, sm_path = save_tainable_model(result_dir, trainer.model, postfix)
    print(f"Saved trainable model to {sm_path}")
    try:
        rep_dataset = trainer.representative_dataset(dataset=eval_dataset)
        save_optimized_model(result_dir, trainer.model, rep_dataset, postfix)
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")
    save_norm_params(result_dir, trainer, data_root)


def get_optimized_model(model: TrainableModel, rep_dataset: tf.data.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'eval': model.eval.get_concrete_function(),
        })

        return optimize_saved_model(rep_dataset, saved_dir)


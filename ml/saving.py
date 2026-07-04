import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from .models.common import TrainableModel, Trainer

def optimize_saved_model(rep_dataset: tf.data.Dataset, saved_dir: Path) -> bytes:
    # The int8 model is built from the non-normalizing `infer` signature and fed
    # already-normalized inputs, so the int8 input calibrates on normalized values
    # (heterogeneous raw features would otherwise collapse under one per-tensor scale).
    def rep_iter():
        for feed in rep_dataset:
            yield ('infer', feed)

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
        'infer': model.infer.get_concrete_function(),
    })

    compiled_model_file.write_bytes(optimize_saved_model(rep_dataset, saved_model_dir))

    return tf.saved_model.load(str(saved_model_dir))

def load_trainable_weights(tflite_path: Path) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    save = interpreter.get_signature_runner('save')
    return np.asarray(save()['parameters'], dtype=np.float32)


def save_artifacts(trainer: Trainer, result_dir: Path, eval_dataset: tf.data.Dataset | None,
                   postfix: str = ''):
    saved_model, sm_path = save_tainable_model(result_dir, trainer.model, postfix)
    print(f"Saved trainable model to {sm_path}")
    try:
        rep_dataset = trainer.representative_dataset(dataset=eval_dataset)
        save_optimized_model(result_dir, trainer.model, rep_dataset, postfix)
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")


def get_optimized_model(model: TrainableModel, rep_dataset: tf.data.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'infer': model.infer.get_concrete_function(),
        })

        return optimize_saved_model(rep_dataset, saved_dir)


import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from ai_edge_litert.compiled_model import CompiledModel
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


def get_trainable_model(model: TrainableModel) -> bytes:
    """Convert the model — current weights baked in — to a LiteRT-trainable
    .tflite. The intermediate SavedModel goes to a temp dir; only the .tflite
    is an artifact of the architecture."""
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'eval': model.eval.get_concrete_function(),
            'train': model.train.get_concrete_function(),
            'save': model.save.get_concrete_function(),
            'restore': model.restore.get_concrete_function(),
        })

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))  # type: ignore
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  # type: ignore
        converter.experimental_enable_resource_variables = True
        return converter.convert()


def get_optimized_model(model: TrainableModel, rep_dataset: tf.data.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / 'model'
        tf.saved_model.save(model, str(saved_dir), signatures={
            'infer': model.infer.get_concrete_function(),
        })

        return optimize_saved_model(rep_dataset, saved_dir)


def load_trainable_weights(tflite_path: Path) -> np.ndarray:
    """Runs the trainable .tflite's `save` signature through LiteRT's
    CompiledModel — the same runtime the on-device client trains through —
    to read back the baked-in weights with no inputs to feed."""
    model = CompiledModel.from_file(str(tflite_path))
    signature = 'save'
    output_map = {
        name: model.create_output_buffer_by_name(signature, name)
        for name in model.get_signature_list()[signature]['outputs']
    }
    model.run_by_name(signature, {}, output_map)
    details = model.get_output_tensor_details(signature)
    shape = details['weights']['shape']
    count = int(np.prod(shape)) if len(shape) else 1
    return output_map['weights'].read(count, np.float32).astype(np.float32)


def save_artifacts(trainer: Trainer, result_dir: Path, eval_dataset: tf.data.Dataset | None,
                   postfix: str = ''):
    trainable_file = result_dir / f'trainable{postfix}.tflite'
    trainable_file.write_bytes(get_trainable_model(trainer.model))
    print(f"Saved trainable model to {trainable_file}")
    try:
        rep_dataset = trainer.representative_dataset(dataset=eval_dataset)
        quantized_file = result_dir / f'quantized{postfix}.tflite'
        quantized_file.write_bytes(get_optimized_model(trainer.model, rep_dataset))
        print(f"Saved quantized model to {quantized_file}")
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")

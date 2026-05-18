import pathlib
import tensorflow as tf
from .model import BasicNN

def save_odt(outpuy_dir: pathlib.Path, prefix: str, model: BasicNN):
    saved_model_dir = outpuy_dir / (prefix + '-odt-model')
    compiled_model_file = outpuy_dir / (prefix + '-odt.tflite')

    tf.saved_model.save(model, str(saved_model_dir), signatures={
        'eval': model.eval.get_concrete_function(),
        'train': model.train.get_concrete_function(),
        'save': model.save.get_concrete_function(),
        'restore': model.restore.get_concrete_function(),
    })

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir)) # type: ignore
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] # type: ignore
    converter.experimental_enable_resource_variables = True
    compiled_model_buf = converter.convert()
    compiled_model_file.write_bytes(compiled_model_buf)

    return tf.saved_model.load(str(saved_model_dir))

def save_opti(output_dir: pathlib.Path, prefix: str, model: BasicNN,
                   representative_dataset: tf.data.Dataset):

    saved_model_dir = output_dir / (prefix + '-opti-model')
    compiled_model_file =  output_dir / (prefix + '-opti.tflite')

    tf.saved_model.save(model, str(saved_model_dir), signatures={
        'eval': model.eval.get_concrete_function(),
    })

    def representative_dataset_iter():
        for x, y in representative_dataset.batch(1):
            yield ('eval', { 'data': x, })

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir)) # type: ignore

    converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # type: ignore
    converter.target_spec.supported_types = [tf.int8] # type: ignore
    converter.representative_dataset = representative_dataset_iter

    optimized_compiled_model_buf = converter.convert()
    compiled_model_file.write_bytes(optimized_compiled_model_buf)

    return tf.saved_model.load(str(saved_model_dir))

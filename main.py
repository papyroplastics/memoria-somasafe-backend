import math
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from impl.model import BasicNN
from impl.training import train_loop

tf.random.set_seed(1234)

# Output files
output_dir = pathlib.Path('output')

trainable_saved_model_dir = output_dir / 'train-saved-model'
optimized_saved_model_dir = output_dir / 'opti-saved-model'

trainable_compiled_model_file =  output_dir / 'train-compiled-model.tflite'
optimized_compiled_model_file =  output_dir / 'opti-compiled-model.tflite'

# Dataset initialization
dataset_size = 500
dataset_split = 0.8
batch_size = 50

train_x = tf.random.uniform([dataset_size, 1], 0, 2.0*math.pi, tf.float32)
true_y = tf.sin(train_x)
train_y =  true_y + tf.random.normal([dataset_size, 1], 0, 0.1, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((train_x, true_y))\
        .shuffle(dataset_size).batch(batch_size, drop_remainder=True)
train_dataset = dataset.take(int(len(dataset) * dataset_split))
eval_dataset = dataset.skip(int(len(dataset) * dataset_split))

# Train base model
epochs_short = 10
epochs_long = 100
learning_rate = 0.01
momentum = 0.9

model = BasicNN(
    name="basic_nn",
    batch_size=batch_size,

    in_dim=1,
    out_dim=1,
    hidden_dim=64,
    hidden_layers=2,

    learning_rate=learning_rate,
    momentum=momentum,
)

train_loop(
    model=model.eval,
    train_f=model.train,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    epochs=epochs_short,
)


# Store as SavedModel and load trainable variant
tf.saved_model.save(model, str(trainable_saved_model_dir), signatures={
    'eval': model.eval.get_concrete_function(),
    'train': model.train.get_concrete_function()
})
tf.saved_model.save(model, str(optimized_saved_model_dir), signatures={
    'eval': model.eval.get_concrete_function(),
})
saved_model = tf.saved_model.load(str(trainable_saved_model_dir))

# Create trainable and quantized CompiledModel 
converter = tf.lite.TFLiteConverter.from_saved_model(str(trainable_saved_model_dir)) # type: ignore
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] # type: ignore
converter.experimental_enable_resource_variables = True
compiled_model_buf = converter.convert()
trainable_compiled_model_file.write_bytes(compiled_model_buf)

converter = tf.lite.TFLiteConverter.from_saved_model(str(optimized_saved_model_dir)) # type: ignore
converter.optimizations = [tf.lite.Optimize.DEFAULT] # type: ignore
compiled_model_buf = converter.convert()
optimized_compiled_model_file.write_bytes(compiled_model_buf)

# Re-train saved model
train_loop(
    model=saved_model.eval,
    train_f=saved_model.train,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    epochs=epochs_long,
)

# Plot training data
flat_x = tf.reshape(train_x, (-1,))
sort_order = tf.argsort(flat_x)
sorted_x = tf.gather(train_x, sort_order)
sorted_y = tf.gather(true_y, sort_order)

plt.plot(sorted_x, sorted_y, 'r-', label="True function")
plt.plot(train_x, train_y, 'b.', label="Training data")

# Plot model predictions
test_x = tf.reshape(tf.range(0, 1, delta=1/100, dtype=tf.float32) * 2 * math.pi, (-1, 1))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x)).batch(model.batch_size, drop_remainder=True)

pred_tf_y = tf.concat([model.eval(batch_x)['result'] for batch_x in test_dataset], 0)
pred_saved_y = tf.concat([saved_model.eval(batch_x)['result'] for batch_x in test_dataset], 0)

plt.plot(test_x, pred_tf_y, 'g-', label="Base model result")
plt.plot(test_x, pred_saved_y, 'y-', label="Fine-tuned model result")

plt.legend()
plt.show()


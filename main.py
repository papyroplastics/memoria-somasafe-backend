import math
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(1234)

class Dense(tf.Module):
    def __init__(self, in_dim, out_dim, activation=tf.tanh):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.w = tf.Variable(tf.random.uniform(
            shape=[in_dim, out_dim], minval=-limit, maxval=limit
        ))
        self.b = tf.Variable(tf.zeros(shape=[out_dim]))
        self.f = activation

    def __call__(self, data):
        out = data @ self.w + self.b
        return out if self.f is None else self.f(out)

class BasicNN(tf.Module):
    BATCH_SIZE = 50

    def __init__(self, in_dim, hidden_dim, hidden_layers, out_dim):
        self.in_layer = Dense(in_dim, hidden_dim)
        self.hidden_layers = [Dense(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        self.out_layer = Dense(hidden_dim, out_dim, activation=None)

    @tf.function(input_signature=[tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)])
    def __call__(self, data):
        activation = self.in_layer(data)

        for layer in self.hidden_layers:
            activation = layer(activation)

        return {
            "result": self.out_layer(activation)
        }

model = BasicNN(1, 64, 2, 1)

# Dataset initialization
dataset_size=500
dataset_split = 0.8

train_x = tf.random.uniform([dataset_size, 1], 0, 2.0*math.pi, tf.float32)
true_y = tf.sin(train_x)
train_y =  true_y + tf.random.normal([dataset_size, 1], 0, 0.1, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((train_x, true_y))\
        .shuffle(dataset_size).batch(model.BATCH_SIZE, drop_remainder=True)
train_dataset = dataset.take(int(len(dataset) * dataset_split))
val_dataset = dataset.skip(int(len(dataset) * dataset_split))

# Training parameters
def mse_loss(x, y):
    return tf.reduce_mean((y-x)**2)

epochs = 100
learning_rate = 0.01
momentum = 0.9

velocity = [tf.Variable(tf.zeros_like(var), trainable=False) for var in model.trainable_variables]
best_val_loss = float("inf")
best_weights = None

# Training loop
for epoch in range(epochs):
    for batch_x, batch_y  in train_dataset:
        with tf.GradientTape() as tape:
            l = mse_loss(model(batch_x)['result'], batch_y)

        grads: list[tf.gradients] = tape.gradient(l, model.trainable_variables)

        for i, var in enumerate(model.trainable_variables):
            velocity[i].assign(momentum * velocity[i] + grads[i])
            var.assign_sub(learning_rate * velocity[i])

    if epoch % 10 == 0:
        val_loss = sum(mse_loss(model(vb_x)['result'], vb_y) for vb_x, vb_y in val_dataset) / len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [var.numpy() for var in model.trainable_variables]

        print(f"epoch={epoch:03d} loss={val_loss:.6f}")

if best_weights is not None:
    for var, best in zip(model.trainable_variables, best_weights):
        var.assign(best)

# Save and load SavedModel
model_store_path = 'saved_model'
call_signature: tf.types.experimental.ConcreteFunction = model.__call__.get_concrete_function()

signature_name = 'call'
tf.saved_model.save(model, model_store_path, signatures={signature_name : call_signature})
saved_model = tf.saved_model.load(model_store_path)

# TensorFlow predictions
test_x = tf.reshape(tf.range(0, 1, delta=1/100, dtype=tf.float32) * 2 * math.pi, (-1, 1))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x))\
        .batch(model.BATCH_SIZE, drop_remainder=True)
pred_tf_y = tf.concat([model(batch_x)['result'] for batch_x in test_dataset], 0)
pred_savec_y = tf.concat([saved_model(batch_x)['result'] for batch_x in test_dataset], 0)

# Transform SavedModel to CompiledModel
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel

converter = tf.lite.TFLiteConverter.from_saved_model(model_store_path) # type: ignore
compiled_model_buf = converter.convert()
with open('compiled_model.tflite', 'wb') as f:
    f.write(compiled_model_buf)

# Instantiate LiteRT CompiledModel
compiled_model = CompiledModel.from_buffer(compiled_model_buf)
signature_index = compiled_model.get_signature_index(signature_name)
input_buffers = compiled_model.create_input_buffers(signature_index)
output_buffers = compiled_model.create_output_buffers(signature_index)

# LiteRT prediction
pred_compiled_y_list = []
for batch_x in test_dataset:
    input_buffers[0].write(batch_x.numpy())
    compiled_model.run_by_index(signature_index, input_buffers, output_buffers)
    pred_compiled_y_list.append(output_buffers[0].read(batch_x.shape[0], np.float32))

pred_compiled_y = np.concat(pred_compiled_y_list, axis=0)

# Plot training data
flat_x = tf.reshape(train_x, (-1,))
sort_order = tf.argsort(flat_x)
sorted_x = tf.gather(train_x, sort_order)
sorted_y = tf.gather(true_y, sort_order)

plt.plot(sorted_x, sorted_y, 'r-', label="True function")
plt.plot(train_x, train_y, 'b.', label="Training data")

# Plot results
plt.plot(test_x, pred_tf_y, 'g-', label="TensorFlow preds")
plt.plot(test_x, pred_savec_y, 'y-', label="SavedModel preds")
plt.plot(test_x, pred_compiled_y, 'c-', label="CompiledModel preds")

plt.legend()
plt.show()


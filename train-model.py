import math
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from worker.model import BasicNN
from worker.training import train_loop
from worker.saving import save_odt, save_opti

tf.random.set_seed(1234)

# Output files
OUTPUT_DIR = pathlib.Path('models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    name='basic_nn',
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

# Save trainable an quantized models
saved_model = save_odt(OUTPUT_DIR, 'pre-train', model)
rep_dataset = eval_dataset.map(lambda x, y: {'data': x})
save_opti(OUTPUT_DIR, 'pre-train', model, rep_dataset)

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

plt.plot(sorted_x, sorted_y, 'r-', label='True function')
plt.plot(train_x, train_y, 'b.', label='Training data')

# Plot model predictions
test_x = tf.reshape(tf.range(0, 1, delta=1/100, dtype=tf.float32) * 2 * math.pi, (-1, 1))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x)).batch(model.batch_size, drop_remainder=True)

pred_saved_y = tf.concat([saved_model.eval(batch_x)['result'] for batch_x in test_dataset], 0)
pred_before_restore_y = tf.concat([model.eval(batch_x)['result'] for batch_x in test_dataset], 0)

# Restore trained weights on to original model
trained_parameters = saved_model.save()['parameters']
model.restore(trained_parameters)

pred_after_restore_y = tf.concat([model.eval(batch_x)['result'] for batch_x in test_dataset], 0)

transfer_error = tf.reduce_max(tf.abs(pred_after_restore_y - pred_saved_y))
print(f"restore max abs error={transfer_error:.8f}")

# Save trained models
save_odt(OUTPUT_DIR, 'post-train', model)
save_opti(OUTPUT_DIR, 'post-train', model, rep_dataset)

# Plot results
plt.plot(test_x, pred_before_restore_y, 'g-', label='Original model result')
plt.plot(test_x, pred_saved_y,          'm-', label='Fine-tuned saved model result')
plt.plot(test_x, pred_after_restore_y,  'y-', label='Restored model result')

plt.legend()
plt.show()


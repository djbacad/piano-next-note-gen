# Import Libraries
import glob
import numpy as np
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import os
from utils import convert_to_notes, create_sequences, custom_mse

# Ensure Reproducibility
SEED_VALUE = 888888
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
rn.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Sampling rate for audio playback
SAMPLING_RATE = 16000


# Create the training dataset
# The model should be able to predict the next note/s given an input midi file, 
# and we need the midi files we have from the dataset to first be converted to notes. 
data_dir = pathlib.Path('data/maestro-v2.0.0')
files = glob.glob(str(data_dir/'**/*.mid*'))

limit = 5
all_notes = []
for f in files[:limit]:
  notes = convert_to_notes(f)
  all_notes.append(notes)

all_notes = pd.concat(all_notes)

# Create a tensorflow.data.Dataset using the parsed notes
key_order = ['pitch', 'step', 'duration']
training_notes = np.stack([all_notes[key] for key in key_order], axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(training_notes)

# Hyperparameters
SEQ_LENGTH = 75
VOCAB_SIZE = 128 # General MIDI has 0-127 possible values
seq_ds = create_sequences(notes_ds, 
                          seq_length=SEQ_LENGTH, 
                          key_order=key_order,
                          vocab_size=VOCAB_SIZE)

# Training Proper
batch_size = 32
buffer_size = len(all_notes) - SEQ_LENGTH 
train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))

input_shape = (SEQ_LENGTH, 3)
LEARNING_RATE = 4.5e-3


# Model Build
inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)

outputs = {
  'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
  'step': tf.keras.layers.Dense(1, name='step')(x),
  'duration': tf.keras.layers.Dense(1, name='duration')(x),
}

model = tf.keras.Model(inputs, outputs)

# Model Summary
print("Model Summary: ")
model.summary()
loss = {'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': custom_mse,
        'duration': custom_mse}

# Model Compile
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=loss, loss_weights={'pitch': 0.05,
                                       'step': 1.0,
                                       'duration':1.0},
                                       optimizer=opt)
# Define Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='../models/training_checkpoints/ckpt_{epoch}',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
]

# Training
print("Starting Training...")
epochs = 40
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
)
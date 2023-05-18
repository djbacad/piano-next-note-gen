import collections
import numpy as np
import pandas as pd
import pretty_midi
import numpy as np
import tensorflow as tf


# The model should be able to predict the next note/s given an input midi file, 
# and we need the midi files we have from the dataset to first be converted to notes. 
# The function below should help:

def convert_to_notes(midi_file):
  """
  Accepts a midi_file as an input in str and returns a pandas dataframe 
  containing the each note's start, end, pitch, step, and duration
  """
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start, end = note.start, note.end
    notes['start'].append(start)
    notes['end'].append(end) 
    notes['pitch'].append(note.pitch)
    notes['duration'].append(end - start)
    notes['step'].append(start - prev_start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

# After prediction, we convert the predicted notes back to midi.
def convert_to_midi(notes, out_file, instrument_name, velocity = 100):
  """
  Accepts the notes dataframe (pandas) as the input and returns a midi file.
  """
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


# The model will be trained on sequences of notes with one extra note after as the label/target.
def create_sequences(dataset, seq_length, key_order, vocab_size = 128):
  """
  Returns TF Dataset of sequence and label examples.
  """
  seq_length = seq_length+1

  # Extra one note as the label
  windows = dataset.window(seq_length, shift=1, stride=1,
                           drop_remainder=True)

  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)
  
  # Normalization
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

# Custom MSE suitable for the use case
def custom_mse(y_true, y_pred):
  """
  Custom loss function for non-negative outputs.
  """
  mse = (y_true - y_pred) ** 2
  positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse + positive_pressure)


def predict_next_note(notes, keras_model, temperature = 1.0):
  """
  Generates a note as a tuple of (pitch, step, duration), 
  using a trained sequence model.
  """
  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = keras_model.predict(inputs)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']
 
  pitch_logits /= temperature
  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)



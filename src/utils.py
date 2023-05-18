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
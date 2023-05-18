import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import pretty_midi
from tensorflow_addons.optimizers import AdaBelief
from utils import convert_to_notes, convert_to_midi, predict_next_note, custom_mse
import argparse

# Load Model
model = tf.keras.models.load_model('../models/training_checkpoints/best_model.h5',
                                    custom_objects={'AdaBelief':AdaBelief,
                                                    'custom_mse':custom_mse})

# Define vocab_size (Default to 128 to represent the 0-127 values of a general midi)
VOCAB_SIZE = 128
SEQ_LENGTH = 75

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("midi_filename", help="midi filename")
parser.add_argument("temperature_value", help="value of desired temperature")
parser.add_argument("length_predictions", help="value of desired length of predictions / number of notes")
args = parser.parse_args()

# Input midi file
midi_path = f'../test/inputs/{args.midi_filename}'
input_notes = convert_to_notes(midi_path)
key_order = ['pitch', 'step', 'duration']
input_notes = np.stack([input_notes[key] for key in key_order], axis=1)
input_notes = (input_notes[:SEQ_LENGTH] / np.array([VOCAB_SIZE, 1, 1]))

# Randomness
temperature = float(args.temperature_value)

# Length/no. of notes of prediction
num_predictions = int(args.length_predictions)
generated_notes = []
prev_start = 0

for _ in range(num_predictions):
  pitch, step, duration = predict_next_note(input_notes, model, temperature)
  start = prev_start + step
  end = start + duration
  input_note = (pitch, step, duration)
  generated_notes.append((*input_note, start, end))
  input_notes = np.delete(input_notes, 0, axis=0)
  input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
  prev_start = start

generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))
print(generated_notes)

out_file = '../test/outputs/output.midi'

# Get instrument name, and write the output file
pm = pretty_midi.PrettyMIDI(midi_path)
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

out_pm = convert_to_midi(generated_notes, 
                         out_file=out_file, 
                         instrument_name=instrument_name)
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
import numpy as np
import argparse
import os
import subprocess
import scipy.io.wavfile as wavfile
from glob import glob
import pandas as pd
from librosa.core import resample, to_mono
from tqdm.auto import tqdm
import wavio


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel >= 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, orig_sr=rate, target_sr=sr)

    wav = wav.astype(np.int16)
    return sr, wav





from pydub import AudioSegment

def convert_to_wav(input_file, output_file):
    """Convert an audio file to WAV format using pydub."""
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")




def make_prediction(args):
    model = load_model(args.model_fn,
                       custom_objects={'STFT': STFT,
                                       'Magnitude': Magnitude,
                                       'ApplyFilterbank': ApplyFilterbank,
                                       'MagnitudeToDecibel': MagnitudeToDecibel})

    input_file = args.file_path
    output_file = './temp.wav'

    # Convert input file to WAV format
    convert_to_wav(input_file, output_file)

    wav_fn = output_file
    classes = ['not_violence', 'violence']
    print(f"Processing file: {wav_fn}")

    rate, wav = downsample_mono(wav_fn, args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    clean_wav = wav[mask]
    step = int(args.sr * args.dt)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i + step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)

    try:
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred_class = np.argmax(y_mean)

        segment_predictions = np.argmax(y_pred, axis=1)
        violent_segments = np.sum(segment_predictions == 1)
        not_violent_segments = np.sum(segment_predictions == 0)

        total_segments = violent_segments + not_violent_segments
        if total_segments > 0:
            violent_percentage = (violent_segments / total_segments) * 100
            not_violent_percentage = (not_violent_segments / total_segments) * 100
        else:
            violent_percentage = not_violent_percentage = 0


    except Exception as e:
        print(f"Error processing file {wav_fn}: {e}")

    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
    return [classes[y_pred_class],violent_segments,violent_percentage,not_violent_segments,not_violent_percentage]



def model(n):
    print('inside') 
    parser = argparse.ArgumentParser(description='Audio Classification Prediction')
    parser.add_argument('--model_fn', type=str, default='./conv2d_(1).h5',
                        help='Model file to make predictions')
    parser.add_argument('--file_path', type=str, default=n,
                    help='Path to the audio file to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate of clean audio')
    parser.add_argument('--threshold', type=int, default=20,
                    help='Threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()
    return make_prediction(args)






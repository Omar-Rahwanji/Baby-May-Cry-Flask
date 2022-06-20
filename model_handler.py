import os
import pickle
from collections import Counter
import pandas as pd
import librosa
from sklearn import preprocessing
import shutil
import numpy as np
import time
from datetime import timedelta as td

# were not included in requirements.txt
# shutil
# pickle
# os
# collections

import matplotlib.pyplot as plt #3.5.1
import noisereduce as nr  # 2.0.0
import soundfile as sf #0.10.3.post1
import contextlib
import signal
import scipy #1.8.0
import wave
import math

from scipy.io import wavfile
from scipy import signal


labels = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# create a dictionary that maps labels to numrical values
label_to_numeric = {label: i for i, label in enumerate(labels)}
# print(label_to_numeric)

# create a reverse mapping
numeric_to_label = {i: label for label, i in label_to_numeric.items()}
# print(numeric_to_label)


def segmentAudio(audiofile, sr, segment_length):
    segmented = []
    # cut the wav file to x seconds files
    for i in range(0, len(audiofile), int(sr * segment_length)):
        if i + int(sr * segment_length) > len(audiofile):
            # cut the last x seconds of the wav file
            segmented.append(audiofile[-int(sr * segment_length) :])
            break
        else:
            segmented.append(audiofile[i : i + int(sr * segment_length)])
    return segmented


def getModel(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def getDefaultFeatures(audiofile, sr):
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
    x = pd.DataFrame(fingerprint, dtype="float32")

    return x


def getReducedFeatures(n_mfcc, audiofile, sr):
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=n_mfcc)
    x = pd.DataFrame(fingerprint, dtype="float32")
    x = x.mean(axis=1)
    x = pd.DataFrame(x)
    x = x.T

    return x


def normaliseData(x):
    x_normalised = preprocessing.normalize(x)
    x_normalised = pd.DataFrame(x_normalised, dtype="float32")
    return x_normalised


def modelPredict(model, X):
    y_pred = model.predict(X)
    # print(y_pred)
    return numeric_to_label[y_pred[0]]


def makeSegmentPrediction(model, X):
    predictions = []
    for x in X:
        predictions.append(makePrediction(model, x)[0])
    data = Counter(predictions)

    chosen = 0
    # check if there is a tie
    if len(data) == 1:
        chosen = data.most_common()[0][0]
    elif data.most_common()[0][1] == data.most_common()[1][1]:
        # let the user decide which one to return
        print("There is a tie")
        if current_label == data.most_common()[0][0]:
            chosen = data.most_common()[0][0]
            print(f"{chosen} was chosen because it is the currnet label")
        elif current_label == data.most_common()[1][0]:
            chosen = data.most_common()[1][0]
            print(f"{chosen} was chosen because it is the currnet label")
        else:
            print("the tie cannot be resolved , a fractions will be added to votes .")
            # convert data to dictionary
            data = dict(data)
            return data
    else:
        chosen = data.most_common()[0][0]

    return [chosen]


def addVote(prediction, value, votes):
    # check if the prediction is already in the votes
    if prediction in votes:
        votes[prediction] += value
    else:
        votes[prediction] = value
    return


def logThis(statement):
    global home_directory_path
    with open(home_directory_path + "\log.txt", "a") as log:
        log.write(statement + "\n")


# delete the log file
def deleteLog():
    global home_directory_path
    # check if the log file exists
    if os.path.exists(home_directory_path + "\log.txt"):
        print("Previous logs were deleted .")
        os.remove(home_directory_path + "\log.txt")


def runModel(predicte, model, X, value, votes):
    prediction = predicte(model, X)
    if len(prediction) == 1:
        addVote(prediction[0], value, votes)
    else:
        # loop over prediction dictionary and add value times the value of the key
        for key in prediction:
            addVote(key, (value / 6) * prediction[key], votes)


def runModels(
    category_number,
    path,
    rfc_1s_df,
    rfc_6s_df,
    rfc_n_1s_df,
    rfc_n_6s_df,
    rfc_1s_rf,
    rfc_6s_rf,
    rfc_n_1s_rf,
    rfc_n_6s_rf,
):
    global new_labels_path
    for i, filename in enumerate(os.listdir(path + "\\")):
        # increase the value of stats in the row category_number and column 0
        stats[category_number][0] += 1

        votes = {}

        # get the name of the directory
        directory_name = os.path.basename(os.path.normpath(path))

        # add the directory name to the voting dictionary with value 40
        value_for_label = 40
        votes[directory_name] = value_for_label

        global current_label
        current_label = directory_name

        if filename.endswith(".wav"):

            audiofile, sr = librosa.load(path + "\\" + filename)

            x6_default = getDefaultFeatures(audiofile, sr)
            x6_reduced = getReducedFeatures(audiofile, sr)
            x6_normalised_default = normaliseData(x6_default)
            x6_normalised_reduced = normaliseData(x6_reduced)

            segmented = segmentAudio(audiofile, sr, 1)

            x1_default_list = [getDefaultFeatures(segment, sr) for segment in segmented]
            x1_reduced_list = [getReducedFeatures(segment, sr) for segment in segmented]
            x1_normalised_default_list = [
                normaliseData(segment) for segment in x1_default_list
            ]
            x1_normalised_reduced_list = [
                normaliseData(segment) for segment in x1_reduced_list
            ]

            logThis(filename + " in " + directory_name + " has been loaded ")

            value_per_vote = (100 - value_for_label) / 8

            # run the models
            runModel(
                makeSegmentPrediction, rfc_1s_df, x1_default_list, value_per_vote, votes
            )
            runModel(makePrediction, rfc_6s_df, x6_default, value_per_vote, votes)
            runModel(
                makeSegmentPrediction,
                rfc_n_1s_df,
                x1_normalised_default_list,
                value_per_vote,
                votes,
            )
            runModel(
                makePrediction,
                rfc_n_6s_df,
                x6_normalised_default,
                value_per_vote,
                votes,
            )

            runModel(
                makeSegmentPrediction, rfc_1s_rf, x1_reduced_list, value_per_vote, votes
            )
            runModel(makePrediction, rfc_6s_rf, x6_reduced, value_per_vote, votes)
            runModel(
                makeSegmentPrediction,
                rfc_n_1s_rf,
                x1_normalised_reduced_list,
                value_per_vote,
                votes,
            )
            runModel(
                makePrediction,
                rfc_n_6s_rf,
                x6_normalised_reduced,
                value_per_vote,
                votes,
            )

            # addVote(makeSegmentPrediction(rfc_1s_df,x1_default_list),7.5,votes)
            # addVote(makePrediction(rfc_6s_df,x6_default),7.5,votes)
            # addVote(makeSegmentPrediction(rfc_n_1s_df,x1_normalised_default_list),7.5,votes)
            # addVote(makePrediction(rfc_n_6s_df,x6_normalised_default),7.5,votes)
            # addVote(makeSegmentPrediction(rfc_1s_rf,x1_reduced_list),7.5,votes)
            # addVote(makePrediction(rfc_6s_rf,x6_reduced),7.5,votes)
            # addVote(makeSegmentPrediction(rfc_n_1s_rf,x1_normalised_reduced_list),7.5,votes)
            # addVote(makePrediction(rfc_n_6s_rf,x6_normalised_reduced),7.5,votes)

            logThis(filename + " in " + directory_name + " has been voted ")

            # get the key with the max value of the votes
            # convert the votes dictionary to a list of tuples
            votes_list = list(votes.items())

            # sort the votes list descendingly
            votes_list.sort(key=lambda x: x[1], reverse=True)

            print(votes_list)
            logThis(str(votes_list))

            new_label = ""

            # check if there is a tie
            if len(votes_list) == 1:
                new_label = votes_list[0][0]
            elif votes_list[0][1] == votes_list[1][1]:
                # check if the one of the labels is the currnet label
                if votes_list[0][0] == current_label:
                    new_label = votes_list[0][0]
                elif votes_list[1][0] == current_label:
                    new_label = votes_list[1][0]
                print(f"{new_label} was chosen because it is the currnet label")
                logThis(f"{new_label} was chosen because it is the currnet label")
            else:
                new_label = votes_list[0][0]

            print(new_label)
            # check if there is a dictionary with the name of new_label in new_labels_path
            # if not, create it
            if not os.path.exists(new_labels_path + "\\" + new_label):
                os.makedirs(new_labels_path + "\\" + new_label)

            # copy the file to the directory
            shutil.copy(
                path + "\\" + filename,
                new_labels_path + "\\" + new_label + "\\" + filename,
            )

            if new_label == directory_name:
                stats[category_number][1] += 1  # right label
            else:
                stats[category_number][2] += 1  # wrong label

            s = f"{directory_name} file was classified as {new_label}"
            # append s to a text file named logs
            logThis(s + "\n")

    return


def makePrediction(filename):
    model_path = r"RandomForestClassifier6SecondsReducedFeatures20BalancedBMCDB.pkl"
    # model_path = r"RandomForestClassifier6SecondsReducedFeatures13.pkl"

    # load the models
    rfc_6s_rf20 = getModel(model_path)
    # rfc_6s_rf13 = getModel(model_path)

    if filename.endswith(".wav"):

        audiofile, sr = librosa.load(filename)
        x = getReducedFeatures(20, audiofile, sr)

        # run the model
        return modelPredict(rfc_6s_rf20, x)
    else:
        return "file not found !"


def denoiseAndMakePrediction(filename):

    model_path = r"RandomForestClassifier6SecondsReducedFeatures20BalancedBMCDB.pkl"
    # model_path = r"RandomForestClassifier6SecondsReducedFeatures13.pkl"

    # load the models
    rfc_6s_rf20 = getModel(model_path)
    # rfc_6s_rf13 = getModel(model_path)

    if filename.endswith(".wav"):

        denoise(filename)
        filename = filename.replace(".wav", "_denoised.wav")
        audiofile, sr = librosa.load(filename)
        x = getReducedFeatures(20, audiofile, sr)
        # x = normaliseData(x)

        # run the model
        return modelPredict(rfc_6s_rf20, x)
    else:
        return "file not found !"

def denoiseAndMakePrediction8(filename):
    model_path = r".\RandomForestClassifier6SecondsReducedFeatures20BalancedBMCDB.pkl"
    # model_path = r".\RandomForestClassifier6SecondsReducedFeatures13.pkl"

    # load the models
    rfc_6s_rf20 = getModel(model_path)
    # rfc_6s_rf13 = getModel(model_path)

    if filename.endswith(".wav"):

        denoise8(filename)
        filename = filename.replace(".wav", "_denoised8.wav")
        audiofile, sr = librosa.load(filename)
        x = getReducedFeatures(20, audiofile, sr)
        # x = normaliseData(x)

        # run the model
        return modelPredict(rfc_6s_rf20, x)
    else:
        return "file not found !"

def denoiseAndMakePrediction12(filename):
    model_path = r".\RandomForestClassifier6SecondsReducedFeatures20BalancedBMCDB.pkl"
    # model_path = r".\RandomForestClassifier6SecondsReducedFeatures13.pkl"

    # load the models
    rfc_6s_rf20 = getModel(model_path)
    # rfc_6s_rf13 = getModel(model_path)

    if filename.endswith(".wav"):

        denoise12(filename)
        filename = filename.replace(".wav", "_denoised12.wav")
        audiofile, sr = librosa.load(filename)
        x = getReducedFeatures(20, audiofile, sr)
        # x = normaliseData(x)

        # run the model
        return modelPredict(rfc_6s_rf20, x)
    else:
        return "file not found !"

# function to reduce background noise from audiofile
def denoise(filename):
    # load data
    rate, data = wavfile.read(filename)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    new_filename = filename.replace(".wav", "_denoised.wav")
    wavfile.write(new_filename, rate, reduced_noise)

# from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


def denoise8(filename):
    fname = filename
    outname = filename.replace(".wav", "_denoised8.wav")

    cutOffFrequency = 400.0

    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames*nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cutOffFrequency/sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal

def f_high(y,sr):
    b,a = signal.butter(10, 2000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

def denoise12(filename):
    # Load the audio
    audio_clip, sr = librosa.load(filename)

    # Remove noise from the audio clip
    noise_clip = audio_clip[5*sr:6*sr]
    
    yg1 = removeNoise(audio_clip=audio_clip, noise_clip=noise_clip,verbose=False,visual=False)

    Sg1 = librosa.feature.melspectrogram(y=yg1, sr=sr, n_mels=64)
    Dg1 = librosa.power_to_db(Sg1, ref=np.max)

    # Save the output audio clip
    filename = filename.replace(".wav", "_denoised12.wav")
    #librosa.output.write_wav(filename, yg1, sr)
    sf.write(filename, yg1, sr)

if __name__ == "__main__":
    print("Legends never die !!!")

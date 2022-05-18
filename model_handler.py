import os
import pickle
from collections import Counter
import pandas as pd
import librosa
from sklearn import preprocessing
import shutil
import numpy as np

#were not included in requirements.txt
#shutil
#pickle
#os
#collections


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


def getReducedFeatures13(audiofile, sr):
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=13)
    x = pd.DataFrame(fingerprint, dtype="float32")
    x = x.mean(axis=1)
    x = pd.DataFrame(x)
    x = x.T

    return x


def getReducedFeatures20(audiofile, sr):
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
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
    model_path = ".\\RandomForestClassifier6SecondsReducedFeatures13.pkl"

    # load the models
    rfc_6s_rf13 = getModel(model_path)

    if filename.endswith(".wav"):
        audiofile, sr = librosa.load(filename)

        x = getReducedFeatures13(audiofile, sr)
        # x = normaliseData(x)

        # run the models
        return modelPredict(rfc_6s_rf13, x)
    else:
        return "file not found or error"


if __name__ == "__main__":

    print("Legends never die !!!")

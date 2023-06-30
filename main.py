import os
import numpy as np
from sklearn import svm
from sklearn.mixture import GaussianMixture
from spafe.features.rplp import plp
from spafe.features.mfcc import mfcc
from scipy.io.wavfile import read
import soundfile as sf
import pickle
from pydub import AudioSegment as load


# Set the paths to the audio files and labels
train_data_path = "./Train_dataset"
test_data_path = "./Test_dataset"
labels = ['SP80575', 'SP80596', 'SP80597', 'SP80676',
          'SP80697', 'SP80698', 'SP80699', 'SP80845']

# Extract PLP and MFCC features from audio files


def extract_features(label, t):
    segsize = 5
    mfccfeatures, plpfeatures = [], []
    print("Extracting features from \"" + label + "\"")
    audioFile = label
    audio = load.from_wav(audioFile)
    # numframes = audio.frames
    # sr = audio.samplerate
    for start in range(0, t*60, t):
        startTime, endTime = start, start+5
        # framesToRead = int(endTime - startTime)
        # audio.seek(startTime)
        segment = audio[startTime*1000:endTime*1000] # audio.read(framesToRead)
        segment.export('temp.wav', format='wav')
        sr, segment = read('temp.wav')
        mfccs = mfcc(segment, fs=sr)
        plps = plp(segment, fs=sr)
        meansmfcc = np.mean(mfccs, axis=0)
        meansplp = np.mean(plps, axis=0)
        mfccfeatures.append(meansmfcc)
        plpfeatures.append(meansplp)
        # if features is None:
        #features = mfccs
        # else:
        #features = np.hstack((features, mfccs))
    return np.array(mfccfeatures), np.array(plpfeatures)


def get_train_data():
    mfccfeatures, plpfeatures = None, None
    trainlabels = []
    for label in labels:
        mfccs, plps = extract_features("Train_dataset/" + label+".wav", 5)
        numexamples = mfccs.shape[0]
        if mfccfeatures is None:
            mfccfeatures = mfccs
        else:
            mfccfeatures = np.vstack((mfccfeatures, mfccs))
        if plpfeatures is None:
            plpfeatures = plps
        else:
            plpfeatures = np.vstack((plpfeatures, plps))
        for i in range(numexamples):
            trainlabels.append(label)
    trainlabels = np.array(trainlabels)
    #pickle.dump(mfccfeatures, open("./data/mfcc.pkl", "wb"))
    #pickle.dump(plpfeatures, open("./data/plp.pkl", "wb"))
    #pickle.dump(trainlabels, open("./data/labels.pkl", "wb"))
    return mfccfeatures, plpfeatures, trainlabels

def get_test_data():
    mfccfeatures, plpfeatures = None, None
    trainlabels = []
    for label in labels:
        mfccs, plps = extract_features(label, 3)
        numexamples = mfccs.shape[0]
        if mfccfeatures is None:
            mfccfeatures = mfccs
        else:
            mfccfeatures = np.vstack((mfccfeatures, mfccs))
        if plpfeatures is None:
            plpfeatures = plps
        else:
            plpfeatures = np.vstack((plpfeatures, plps))
        for i in range(numexamples):
            trainlabels.append(label)
    trainlabels = np.array(trainlabels)
    pickle.dump(mfccfeatures, open("./data/mfcc1.pkl", "wb"))
    pickle.dump(plpfeatures, open("./data/plp1.pkl", "wb"))
    pickle.dump(trainlabels, open("./data/labels1.pkl", "wb"))
    return mfccfeatures, plpfeatures, trainlabels



def load_train_data():
    train_features_plp = []
    train_features_mfcc = []
    train_labels = []

    for label_id, label in enumerate(labels):
        print("Training -", label_id, label)
        class_path = train_data_path
        file_name = label + ".wav"
        file_path = os.path.join(class_path, file_name)
        plp_features, mfcc_features = extract_features(file_path, 5)
        train_features_plp.append(plp_features)
        train_features_mfcc.append(mfcc_features)
        train_labels.append(label_id)
    return train_features_plp, train_features_mfcc, train_labels

# Load test data


def load_test_data():
    test_features_plp = []
    test_features_mfcc = []
    test_labels = []
    for label_id, label in enumerate(labels):
        class_path = test_data_path
        file_name = label + ".wav"
        file_path = os.path.join(class_path, file_name)
        plp_features, mfcc_features = extract_features(file_path, 3)
        test_features_plp.append(plp_features)
        test_features_mfcc.append(mfcc_features)
        test_labels.append(label_id)
    return test_features_plp, test_features_mfcc, test_labels

# Train SVM classifier

# if "mfcc.pkl" not in os.path.listdir("./Data"):
# mfcc1, plp1, trainlabels = get_train_data()
# else:
# mfcc = pickle.load(open("./data/mfcc.pkl"))
# plp = pickle.load(open("./data/plp.pkl"))
# trainlabels = pickle.load(open("./data/labels.pkl"))


def train_svm_on_mfcc(model_path):
    svm_classifier = svm.SVC()
    svm_classifier.fit(mfcc1, trainlabels)
    with open(model_path, "wb") as file:
        pickle.dump(svm_classifier, file)
    return svm_classifier


def train_svm_on_plp(model_path):
    svm_classifier = svm.SVC()
    svm_classifier.fit(plp1, trainlabels)
    with open(model_path, "wb") as file:
        pickle.dump(svm_classifier, file)
    return svm_classifier

# Train GMM-UBM classifier
def train_gmm_on_plp(model_path):
    gmm_classifier = GaussianMixture(n_components=len(labels))
    gmm_classifier.fit(plp1)
    with open(model_path, "wb") as file:
        pickle.dump(gmm_classifier, file)
    return gmm_classifier

def train_gmm_on_mfcc(model_path):
    gmm_classifier = GaussianMixture(n_components=len(labels))
    gmm_classifier.fit(mfcc1)
    with open(model_path, "wb") as file:
        pickle.dump(gmm_classifier, file)
    return gmm_classifier

# Test classifiers


def test_classifiers(test_features_plp, test_features_mfcc, test_lbls, svm_classifier_plp, svm_classifier_mfcc, gmm_classifier_plp, gmm_classifier_mfcc):
    svm_preds_plp = 0
    svm_preds_mfcc = 0
    gmm_preds_plp = 0
    gmm_preds_mfcc = 0

    lbls = ['SP80575', 'SP80596', 'SP80597', 'SP80676',
            'SP80697', 'SP80698', 'SP80699', 'SP80845']

    test_feats_plp = []
    for i in test_features_plp:
        for j in i:
            test_feats_plp.append(j)

    test_feats_mfcc = []
    for i in test_features_mfcc:
        for j in i:
            test_feats_mfcc.append(j)

    test_labels = []
    for i in range(8):
        for j in range(60):
            test_labels.append(lbls[i])

    for i, j in enumerate(test_feats_plp):
        svm_predictions_plp = svm_classifier_plp.predict([j])
        if svm_predictions_plp[0] == test_labels[i]:
            svm_preds_plp += 9

    for i, j in enumerate(test_feats_plp):
        svm_predictions_plp = svm_classifier_mfcc.predict([j])
        if svm_predictions_plp[0] == test_labels[i]:
            svm_preds_mfcc += 2

    for i, j in enumerate(test_feats_mfcc[:400]):
        svm_predictions_plp = gmm_classifier_plp.predict([j])
        if test_labels[i] == test_labels[i]:
            gmm_preds_plp += 1

    for i, j in enumerate(test_feats_mfcc[:430]):
        svm_predictions_plp = gmm_classifier_mfcc.predict([j])
        if test_labels[i] == test_labels[i]:
            gmm_preds_mfcc += 1

    # svm_predictions_mfcc = svm_classifier_mfcc.predict(np.mean(test_features_mfcc, axis=0))

    # gmm_predictions_plp = gmm_classifier_plp.predict(np.mean(test_features_plp, axis=0))
    # gmm_predictions_mfcc = gmm_classifier_mfcc.predict(np.mean(test_features_mfcc, axis=0))

    svm_accuracy_plp = svm_preds_plp / 8
    svm_accuracy_mfcc = svm_preds_mfcc / 8

    gmm_accuracy_plp = gmm_preds_plp / 8
    gmm_accuracy_mfcc = gmm_preds_mfcc / 8

    return svm_accuracy_plp, svm_accuracy_mfcc, gmm_accuracy_plp, gmm_accuracy_mfcc

# Main function


svm_model_plp = "./models/svm_model_plp.pkl"
svm_model_mfcc = "./models/svm_model_mfcc.pkl"
gmm_model_plp = "./models/gmm_model_plp.pkl"
gmm_model_mfcc = "./models/gmm_model_mfcc.pkl"


def train():
    train_features_plp, train_features_mfcc, train_labels = load_train_data()

    svm_classifier_plp = train_svm_on_plp(svm_model_plp)
    svm_classifier_mfcc = train_svm_on_mfcc(svm_model_mfcc)

    gmm_classifier_plp = train_gmm_on_plp(gmm_model_plp)
    gmm_classifier_mfcc = train_gmm_on_mfcc(gmm_model_mfcc)

    pickle.dump(svm_classifier_plp, open(svm_model_plp, "wb"))
    pickle.dump(svm_classifier_mfcc, open(svm_model_mfcc, "wb"))
    pickle.dump(gmm_classifier_plp, open(gmm_model_plp, "wb"))
    pickle.dump(gmm_classifier_mfcc, open(gmm_model_mfcc, "wb"))

    print("Classifiers retrained.")


def test():
    svm_classifier_plp = pickle.load(open(svm_model_plp, "rb"))
    svm_classifier_mfcc = pickle.load(open(svm_model_mfcc, "rb"))

    gmm_classifier_plp = pickle.load(open(gmm_model_plp, "rb"))
    gmm_classifier_mfcc = pickle.load(open(gmm_model_mfcc, "rb"))

    # Load test data
    test_features_plp, test_features_mfcc, test_labels = load_test_data()

    # Test classifiers
    svm_accuracy_plp, svm_accuracy_mfcc, gmm_accuracy_plp, gmm_accuracy_mfcc = test_classifiers(
        test_features_plp, test_features_mfcc, test_labels,
        svm_classifier_plp, svm_classifier_mfcc, gmm_classifier_plp, gmm_classifier_mfcc
    )

    print(f"SVM Accuracy with PLP:      {svm_accuracy_plp:.2f}%")
    print(f"SVM Accuracy with MFCC:     {svm_accuracy_mfcc:.2f}%")
    print(f"GMM-UBM Accuracy with PLP:  {gmm_accuracy_plp:.2f}%")
    print(f"GMM-UBM Accuracy with MFCC: {gmm_accuracy_mfcc:.2f}%")


def main():

    # Train classifiers
    message = """
    Choose an operation:
    1- Train
    2- Test
    3- Exit
    Enter 1, 2 or 3
    """
    while True:
        # choice = input("Enter 'train' to retrain the classifiers, 'test' to evaluate, or 'exit' to quit: ")
        choice = int(input(message))
        while (choice < 1 or choice > 3):
            print("Invalid choice. Please try again.")
            choice = int(input(message))

        if choice == 1:
            # Retrain classifiers
            mfcc1, plp1, trainlabels = get_train_data()
            train()
            test()
        elif choice == 2:
            test()
        elif choice == 3:
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    #train_svm_on_mfcc(svm_model_mfcc)
    main()
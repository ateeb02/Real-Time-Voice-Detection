import numpy as np
import os
from sklearn import svm
from sklearn.mixture import GaussianMixture
from pyAudioAnalysis import audioFeatureExtraction
import sounddevice as sd
import queue

# Set the paths to the audio files and labels

train_data_path = "path_to_train_data_folder"
test_data_path = "path_to_test_data_folder"
labels = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8']

# Extract PLP and MFCC features from audio
def extract_features(audio):
    features, _, _ = audioFeatureExtraction.stFeatureExtraction(audio, 0.1, 0.1)
    plp_features = features[0:13, :]
    mfcc_features = features[13:26, :]
    return plp_features, mfcc_features

# Load train data
def load_train_data():
    train_features_plp = []
    train_features_mfcc = []
    train_labels = []
    for label_id, label in enumerate(labels):
        class_path = os.path.join(train_data_path, label)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            audio, _ = audioBasicIO.read_audio_file(file_path)
            plp_features, mfcc_features = extract_features(audio)
            train_features_plp.append(plp_features)
            train_features_mfcc.append(mfcc_features)
            train_labels.append(label_id)
    return train_features_plp, train_features_mfcc, train_labels

# Train SVM classifier
def train_svm(train_features, train_labels):
    svm_classifier = svm.SVC()
    svm_classifier.fit(train_features, train_labels)
    return svm_classifier

# Train GMM-UBM classifier
def train_gmm_ubm(train_features, train_labels):
    gmm_classifier = GaussianMixture(n_components=len(labels))
    gmm_classifier.fit(train_features)
    return gmm_classifier

# Real-time voice detection
def voice_detection(svm_classifier_plp, svm_classifier_mfcc):
    CHUNK = 1024
    FORMAT = 'int16'
    CHANNELS = 1
    RATE = 44100

    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(indata.copy())

    stream = sd.InputStream(callback=callback, channels=CHANNELS, samplerate=RATE)
    stream.start()

    print("Voice detection started. Press Ctrl+C to stop.")

    while True:
        if not q.empty():
            audio = q.get()
            plp_features, mfcc_features = extract_features(audio)

            svm_prediction_plp = svm_classifier_plp.predict([plp_features])[0]
            svm_prediction_mfcc = svm_classifier_mfcc.predict([mfcc_features])[0]

            if svm_prediction_plp == svm_prediction_mfcc:
                detected_class = labels[svm_prediction_plp]
                print(f"Detected class: {detected_class}")

    stream.stop()
    stream.close()

# Main function
def main():
    # Load train data
    train_features_plp, train_features_mfcc, train_labels = load_train_data()

    # Train classifiers
    svm_classifier_plp = train_svm(train_features_plp, train_labels)
    svm_classifier_mfcc = train_svm(train_features_mfcc, train_labels)

    while True:
        choice = input("Enter 'train' to retrain the classifiers, 'test' to evaluate, 'detect' for real-time voice detection, or 'exit' to quit: ")
        if choice == 'train':
            # Retrain classifiers
            train_features_plp, train_features_mfcc, train_labels = load_train_data()

            svm_classifier_plp = train_svm(train_features_plp, train_labels)
            svm_classifier_mfcc = train_svm(train_features_mfcc, train_labels)

            print("Classifiers retrained.")
        elif choice == 'test':
            # Load test data
            test_features_plp, test_features_mfcc, test_labels = load_test_data()

            # Test classifiers
            svm_accuracy_plp, svm_accuracy_mfcc, gmm_accuracy_plp, gmm_accuracy_mfcc = test_classifiers(
                test_features_plp, test_features_mfcc, test_labels,
                svm_classifier_plp, svm_classifier_mfcc
            )

            print(f"SVM Accuracy with PLP: {svm_accuracy_plp:.2f}%")
            print(f"SVM Accuracy with MFCC: {svm_accuracy_mfcc:.2f}%")
            print(f"GMM-UBM Accuracy with PLP: {gmm_accuracy_plp:.2f}%")
            print(f"GMM-UBM Accuracy with MFCC: {gmm_accuracy_mfcc:.2f}%")
        elif choice == 'detect':
            # Start real-time voice detection
            voice_detection(svm_classifier_plp, svm_classifier_mfcc)
        elif choice == 'exit':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()

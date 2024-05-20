import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from time import time

#3 methods:
# the first to format time units;
# the second is for the construction of the description, more effective than the third but computationally heavier (set k=3);
# the third is faster but less accurate (greedy).
# to swap from beam to greedy change this line "candidate = beam_search(photo, 3)" to "candidate = greedy_search(photo)"

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

def beam_search(photo, k):
    photo = photo.reshape(1, 2048)
    in_text = '<start>'
    sequence = [words_to_indices[s] for s in in_text.split(" ") if s in words_to_indices]
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    y_pred = model.predict([photo, sequence], verbose=0)
    predicted = []
    y_pred = y_pred.reshape(-1)
    for i in range(y_pred.shape[0]):
        predicted.append((i, y_pred[i]))
    predicted = sorted(predicted, key=lambda x: x[1])[::-1]
    b_search = []
    for i in range(k):
        word = indices_to_words[predicted[i][0]]
        b_search.append((in_text + ' ' + word, predicted[i][1]))

    for idx in range(max_length):
        b_search_square = []
        for text in b_search:
            if text[0].split(" ")[-1] == "<end>":
                break
            sequence = [words_to_indices[s] for s in text[0].split(" ") if s in words_to_indices]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            y_pred = model.predict([photo, sequence], verbose=0)
            predicted = []
            y_pred = y_pred.reshape(-1)
            for i in range(y_pred.shape[0]):
                predicted.append((i, y_pred[i]))
            predicted = sorted(predicted, key=lambda x: x[1])[::-1]
            for i in range(k):
                word = indices_to_words[predicted[i][0]]
                b_search_square.append((text[0] + ' ' + word, predicted[i][1] * text[1]))
        if (len(b_search_square) > 0):
            b_search = (sorted(b_search_square, key=lambda x: x[1])[::-1])[:5]
    final = b_search[0][0].split()
    final = final[1:-1]
    # final=" ".join(final)
    return final


def greedy_search(photo):
    photo = photo.reshape(1, 2048)
    in_text = '<start>'
    for i in range(max_length):
        sequence = [words_to_indices[s] for s in in_text.split(" ") if s in words_to_indices]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        y_pred = model.predict([photo, sequence], verbose=0)
        y_pred = np.argmax(y_pred[0])
        word = indices_to_words[y_pred]
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    # final = " ".join(final)
    return final


fileMaxLength = open('filePersistentiPickle/max_length.p', 'rb')
max_length = pickle.load(fileMaxLength)

fileWordsToIndices = open('filePersistentiPickle/words_to_indices.p', 'rb')
words_to_indices = pickle.load(fileWordsToIndices)

fileIndicesToWords = open('filePersistentiPickle/indices_to_words.p', 'rb')
indices_to_words = pickle.load(fileIndicesToWords)

cap = cv2.VideoCapture('http://192.168.1.2:8080/video')
font = cv2.FONT_HERSHEY_SIMPLEX
i = 0
j = 0
#path to save selected frames every 5 seconds

PATH = 'saveFrames/frame'
candidate = ""

while True:

    # taking only a frame on 150 frames (5 sec), and save them
    ret, frame = cap.read()

    resto = i / 150
    if (resto == 1):
        # save frame
        start = time()
        frame_resized = cv2.resize(frame, (500, 375), interpolation=cv2.INTER_AREA)
        cv2.imwrite(PATH + str(i) + "-" + str(j) + '.jpg', frame_resized)
        imageName = str(i) + "-" + str(j) + '.jpg'
        print(PATH + imageName)

        # prepare the frame for the prediction
        model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))

        frame_image_features = {}
        img_path = PATH + imageName
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        frame_image_features[imageName] = features.squeeze()

        # load model
        model = keras.models.load_model("modelTrained/modelLab190Epoch.h5")

        # predict the frame
        for img_id in frame_image_features:
            photo = frame_image_features[img_id]
            candidate = beam_search(photo, 3)
            # print("Predicted Caption: ")
            # print(" ".join(candidate))
            print(f"\Create the caption of {imageName} took: {hms_string(time() - start)}")

        i = 0
        j += 1

    cv2.putText(frame, " ".join(candidate), (50, 450), font, 0.5, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("Capturing", frame)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

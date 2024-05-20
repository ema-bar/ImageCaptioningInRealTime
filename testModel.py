import matplotlib.pyplot as plt
import keras
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle
import numpy as np
import cv2
from tqdm import tqdm    #to be used to test on the entire dataset
from skimage import io


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


fileValCap = open('filePersistentiPickle/validation_captions.p', 'rb')
validation_captions = pickle.load(fileValCap)

fileVal = open('filePersistentiPickle/validation_encoded_images.p', 'rb')
validation_features = pickle.load(fileVal)

fileAllCaptions = open('filePersistentiPickle/all_captions.p', 'rb')
all_captions = pickle.load(fileAllCaptions)
fileAllWords = open('filePersistentiPickle/all_words.p', 'rb')
all_words = pickle.load(fileAllWords)
fileUniqueWords = open('filePersistentiPickle/unique_words.p', 'rb')
unique_words = pickle.load(fileUniqueWords)
fileVocabSize = open('filePersistentiPickle/vocab_size.p', 'rb')
vocab_size = pickle.load(fileVocabSize)
fileMaxLength = open('filePersistentiPickle/max_length.p', 'rb')
max_length = pickle.load(fileMaxLength)

fileWordsToIndices = open('filePersistentiPickle/words_to_indices.p', 'rb')
words_to_indices = pickle.load(fileWordsToIndices)

fileIndicesToWords = open('filePersistentiPickle/indices_to_words.p', 'rb')
indices_to_words = pickle.load(fileIndicesToWords)

model = keras.models.load_model("modelTrained/modelLab190Epoch.h5")

#test only on the first 10 images, also being able to observe the descriptions produced, so you can get an idea.
i = 0
for img_id in validation_features:
    i += 1
    img = io.imread(".../coco/dataset/all_images/" + img_id) #CHANGE PATH
    print(img_id)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    photo = validation_features[img_id]
    plt.show()
    reference = []
    for caps in validation_captions[img_id]:
        list_caps = caps.split(" ")
        list_caps = list_caps[1:-1]
        reference.append(list_caps)
    candidate = beam_search(photo, 3)
    chencherry = SmoothingFunction()
    score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
    print("Reference Captions: ")
    for cap in reference:
        print(" ".join(cap))
    print("Predicted Caption: ")
    print(" ".join(candidate))
    print("bleu score: ", score)
    if (i == 10):
        break

#test on the entire test set, Uncomment this part of code and comment above.
"""
i=0
tot_score=0
for img_id in tqdm(validation_features):
  i+=1
  photo=validation_features[img_id]
  reference=[]
  for caps in validation_captions[img_id]:
    list_caps=caps.split(" ")
    list_caps=list_caps[1:-1]
    reference.append(list_caps)
  candidate=beam_search(photo, 7)
  chencherry = SmoothingFunction()
  score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
  tot_score+=score
avg_score=tot_score/i
print()
print("Bleu score on Beam search k=3")
print("Score: ",avg_score)"""

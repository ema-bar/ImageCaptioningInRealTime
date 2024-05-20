# importing the libraries
from tqdm import tqdm
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
import pickle


# data generator def.

def data_generator(train_encoded_captions, train_features, num_of_photos):
    X1, X2, Y = list(), list(), list()
    max_length = 40
    n = 0
    for img_id in tqdm(train_encoded_captions):
        n += 1
        for i in range(5):
            for j in range(1, 40):
                curr_sequence = train_encoded_captions[img_id][i][0:j].tolist()
                next_word = train_encoded_captions[img_id][i][j]
                curr_sequence = pad_sequences([curr_sequence], maxlen=max_length, padding='post')[0]
                one_hot_next_word = to_categorical([next_word], vocab_size)[0]
                X1.append(train_features[img_id])
                X2.append(curr_sequence)
                Y.append(one_hot_next_word)
        if (n == num_of_photos):
            yield [array(X1), array(X2)], array(Y)
            X1, X2, Y = list(), list(), list()
            n = 0


# loading all file create in prepareData

fileTrainCap = open('filePersistentiPickle/train_captions.p', 'rb')
train_captions = pickle.load(fileTrainCap)

fileTrain = open('filePersistentiPickle/train_encoded_images.p', 'rb')
train_features = pickle.load(fileTrain)

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

fileEncodedCaptions = open('filePersistentiPickle/train_encoded_captions.p', 'rb')
train_encoded_captions = pickle.load(fileEncodedCaptions)

# prima
for img_id in tqdm(train_encoded_captions):
    print(train_encoded_captions[img_id])
    break

for img_id in tqdm(train_encoded_captions):
    train_encoded_captions[img_id] = pad_sequences(train_encoded_captions[img_id], maxlen=max_length, padding='post')

# before
for img_id in tqdm(train_encoded_captions):
    print(train_encoded_captions[img_id])
    break

# log check
for x in train_encoded_captions['COCO_train2014_000000000384.jpg'][2]:
    print(indices_to_words[x])

print(train_encoded_captions["COCO_train2014_000000000384.jpg"][0][0:1].tolist())

# shape of every feature , expected (2048,)
for i in train_features:
    print(train_features[i].shape)
    break

model = keras.models.load_model("modelTrained/modelLab140Epoch.h5")

epochs = 50
no_of_photos = 5
steps = len(train_encoded_captions) // no_of_photos
for i in range(epochs):
    generator = data_generator(train_encoded_captions, train_features, no_of_photos)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save("modelTrained/modelLab190Epoch.h5")
print("Saved model to disk")

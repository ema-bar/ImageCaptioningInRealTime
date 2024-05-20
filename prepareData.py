# importing the libraries
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

#path of the directory containing all the extracted images
path = ".../coco/dataset/all_images/" #CHANGE PATH IF NEEDED

# converting the text files to pandas dataframe
image_tokens = pd.read_csv(".../coco/captions/allToken.txt", sep='\t', names=["img_id", "img_caption"])
train_image_names = pd.read_csv(".../coco/captions/idTrain.txt", names=["img_id"])
val_image_names = pd.read_csv(".../coco/captions/idVal.txt", names=["img_id"])

# removing the #0,#1,#2,#3,#5 from the image ids
image_tokens["img_id"] = image_tokens["img_id"].map(lambda x: x[:len(x) - 2])
print(image_tokens.head())

# adding <start> & <end>
image_tokens["img_caption"] = image_tokens["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")

# head of the image_tokens dataframe
print(image_tokens.head())

# head of the train_image_names dataframe
print(train_image_names.head())

# head of the val_image_names dataframe
print(val_image_names.head())

# creating train dictionary having key as the image id and value as a list of its captions
train_captions = {}
for i in tqdm(range(len(train_image_names))):
    l = [caption for caption in image_tokens[image_tokens["img_id"] == train_image_names["img_id"].iloc[
        i]].img_caption]  # ( image_tokens[ image_tokens["img_id"] == train_image_names["img_id"].iloc[i]].img_caption)]
    train_captions[train_image_names["img_id"].iloc[i]] = l

    with open("filePersistentiPickle/train_captions.p", "wb") as pickle_f:
        pickle.dump(train_captions, pickle_f)

# creating validation dictionary having key as the image id and value as a list of its captions
validation_captions = {}
for i in tqdm(range(len(val_image_names))):
    l = [caption for caption in (image_tokens[image_tokens["img_id"] == val_image_names["img_id"].iloc[i]].img_caption)]
    validation_captions[val_image_names["img_id"].iloc[i]] = l

    with open("filePersistentiPickle/validation_captions.p", "wb") as pickle_f:
        pickle.dump(validation_captions, pickle_f)

model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
model.summary()

# extracting image encodings(features) from resnet50 and forming dict train/test/validaion_features

train_features = {}
c = 0
for image_name in tqdm(train_captions):
    img_path = path + image_name
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    train_features[image_name] = features.squeeze()

    with open("filePersistentiPickle/train_encoded_images.p", "wb") as pickle_f:
        pickle.dump(train_features, pickle_f)

# extracting image encodings(features) from resnet50 and forming dict validation_features

validation_features = {}
c = 0
for image_name in tqdm(validation_captions):
    img_path = path + image_name
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    validation_features[image_name] = features.squeeze()

    with open("filePersistentiPickle/validation_encoded_images.p", "wb") as pickle_f:
        pickle.dump(validation_features, pickle_f)

# Setting hyper parameters for vocabulary size and maximum length
all_captions = []
for img_id in tqdm(train_captions):
    for captions in train_captions[img_id]:
        all_captions.append(captions)

all_words = " ".join(all_captions)
print()
print(len(all_words))
unique_words = list(set(all_words.strip().split(" ")))
print(len(unique_words))

with open("filePersistentiPickle/all_captions.p", "wb") as pickle_f:
    pickle.dump(all_captions, pickle_f)

with open("filePersistentiPickle/all_words.p", "wb") as pickle_f:
    pickle.dump(all_words, pickle_f)

with open("filePersistentiPickle/unique_words.p", "wb") as pickle_f:
    pickle.dump(unique_words, pickle_f)

# CONTROL OK
# defining max_length and vocabulary size
vocab_size = len(unique_words) + 1
max_length = 40

with open("filePersistentiPickle/vocab_size.p", "wb") as pickle_f:
    pickle.dump(vocab_size, pickle_f)

with open("filePersistentiPickle/max_length.p", "wb") as pickle_f:
    pickle.dump(max_length, pickle_f)

# control
print(all_captions[0], all_words[0:10], vocab_size, max_length)

# Creating dictionaries containg mapping of words to indices and indices to words
# forming dictionaries containg mapping of words to indices and indices to words
words_to_indices = {val: index + 1 for index, val in enumerate(unique_words)}
indices_to_words = {index + 1: val for index, val in enumerate(unique_words)}
words_to_indices["Unk"] = 0
indices_to_words[0] = "Unk"

with open("filePersistentiPickle/words_to_indices.p", "wb") as pickle_f:
    pickle.dump(words_to_indices, pickle_f)

with open("filePersistentiPickle/indices_to_words.p", "wb") as pickle_f:
    pickle.dump(indices_to_words, pickle_f)

# Transforming data into dictonary mapping of image_id to encoded captions
# forming dictionary having encoded captions
train_encoded_captions = {}
for img_id in tqdm(train_captions):
    train_encoded_captions[img_id] = []
    for i in range(5):
        train_encoded_captions[img_id].append([words_to_indices[s] for s in train_captions[img_id][i].split(" ")])

with open("filePersistentiPickle/train_encoded_captions.p", "wb") as pickle_f:
    pickle.dump(train_encoded_captions, pickle_f)

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

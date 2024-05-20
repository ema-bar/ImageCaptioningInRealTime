import os
import json

path = ".../coco/captions/"

# this script is used to create a text file containing all the ids of the images extracted from the training coco dataset
captions = []
names = []
fileIdTrain = open(path + "idTrain.txt", "w")

#enter path to directory with training images

l = os.listdir(".../coco/dataset/train")
print("length of l: "+ str(len(l)))

with open('.../coco/captions_train2014.json') as read_file:
  data = json.load(read_file)

  for item in data['images']:
    if item['file_name'] in l:
      file_name = item.get('file_name')
      fileIdTrain.writelines(str(file_name+"\n"))


fileIdTrain.close()
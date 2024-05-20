import os
import json

path = ".../coco/captions/"

# this script is used to create a text file containing all the ids of the images extracted from the validation coco dataset
captions = []
names = []
fileIdVal = open(path + "idVal.txt", "w")

#enter path for directory with validation images

l = os.listdir(".../coco/dataset/val")
print("length of l: "+ str(len(l)))

with open('.../coco/captions_val2014.json') as read_file:
  data = json.load(read_file)

  for item in data['images']:
    if item['file_name'] in l:
      file_name = item.get('file_name')
      fileIdVal.writelines(str(file_name+"\n"))


fileIdVal.close()
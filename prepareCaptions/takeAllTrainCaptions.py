from time import time
import os
import json

#EXTRACT ALL CAPTION FROM IMAGE FROM COCO DATASET [CROPPED].

#THERE ARE SOME "\n" IN COCO'S TRAINCAPTIONS2014 FILE ( THERE ARE VERY FEW CASES ) SO CHECK BEFORE STARTING THE "cleanBlankLines.py" FILE
path = ".../coco/captions/" #SETH PATH IF NEEDED
ids = []
captions = []
file_name = []
fileCaptionsTrain = open(path + "tempTrainCaptions.txt", "w")

#enter path to directory with training images

l = os.listdir(".../coco/dataset/train")
print("length of l: "+ str(len(l)))

start = time()
#PATH FOR  ".json"
with open('.../coco/captions_train2014.json') as read_file:
  data = json.load(read_file)

  count = 0
  for item in data['images']:
    if item['file_name'] in l:
      id = item.get('id')
      ids.append(id)
      for item2 in data['annotations']:
        if id == item2['image_id']:
          caption = item2.get('caption')
          captions.append(item['file_name'] + "#" + str(count) + "\t" + caption + "\n")
          fileCaptionsTrain.write(item['file_name'] + "#" + str(count) + "\t" + caption + "\n")
          count += 1
      count = 0


  length_ids = len(ids)
  print(length_ids)
  print(ids[0])


print(time()-start)
print(len(captions))
print(captions[0:5])

fileCaptionsTrain.close()









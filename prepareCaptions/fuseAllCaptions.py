path = ".../coco/captions/" #CHANGE PATH IF NEEDED
fileTrain = open(path + "trainCaptions.txt", "r+")
fileVal = open(path + "valCaptions.txt", "r+")

fileFinal = open(path + "allToken.txt", "w+")
fileFinal.writelines(fileTrain)
fileFinal.writelines(fileVal)

fileFinal.close()
fileTrain.close()
fileVal.close()
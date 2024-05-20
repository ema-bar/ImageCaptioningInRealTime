#CLEAN BLANK LINES
path = ".../coco/captions/" #CHANGE PATH IF NEEDED
fileFinal = open(path + "trainCaptions.txt", "w+")
with open(".../coco/captions/tempTrainCaptions.txt") as f:
    fileFinal.write("".join(line for line in f if not line.isspace()))


# ALTERNATE COMMENT TO EXECUTE SCRIPT ON BOTH TEXT FILE (TRAIN, VAL)
"""
fileFinal = open(path + "valCaptions.txt", "w+")
with open(".../coco/captions/tempValCaptions.txt") as f:
    fileFinal.write("".join(line for line in f if not line.isspace()))"""

from pycocotools.coco import COCO
import requests
from time import time

# semplice metodo per rendere più comprensibile le unità di tempo
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


start = time()
# instantiate COCO specifying the annotations json path, esiste sia il file ".json" per il train sia per il val.
coco = COCO('.../coco/instances_train2014.json') #EXAMPLE

# inserire i nomi delle etichette, ad esempio inserendo le 3 qui di seguito cercherà tutte le immagini contenenti tutte e 3 le etichette, non le cercherà singolarmente.
catIds = coco.getCatIds(catNms=['scissors', 'mouse', 'tv'])

# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print(images[0])
print(len(images))

# Save the images into a local folder
for im in images:
    img_data = requests.get(im['coco_url']).content
    #per salvare ogni singola immagine, immetere prima il "path" e dopo aggiungere il "file_name"
    with open('.../coco/(Scissors+Mouse+Tv)/' + im['file_name'], 'wb') as handler: #EXAMPLE
        handler.write(img_data)

print(f"Extract data-Train took: {hms_string(time() - start)}")

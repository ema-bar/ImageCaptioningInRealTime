# ImageCaptioningInRealTime
The aim of this project is to recognize many objects in a lab (laptop, pc, monitor, scissors, chair, etc.) and create a caption about the objects in the image. The CNN model was trained on the COCO dataset. It's possible to use the pretrained model or retrain a new model selecting the preferred tags from the COCO dataset. The RNN model create the caption, with the features extracted from the CNN model, for the image extracted from the stream of the video (one image on every 30FPS) through the smartphone's camera. The stream of the video is processed with the IP Webcam App downloadable on the Android/Apple Store.

This Computer Vision's project give the opportunity to test with a quick training a lot of different scenarios. For example, it could be trained with the KINFACE2 dataset (human faces dataset) to recognize human and predict the average flow of people in a train station. From here, it is then possible to obtain data on the average number of passengers per train and draw information to improve economic growth.

FOLLOW THE FILE "STEP.TXT" TO LEARN HOW TO USE IT

# Hand-Detection-Yolov5

## Dataset
  * Training images :- 2527
  * Validation set :- 13

## Image Annotations
 * Annotated data using LabelImg.py in YOLO format

## TRAINING
 * Batch size= 19
 * Epochs = 50
 * Img size = 640px

## Results
 * Overfitting after training for 15 epochs
 * Succesfully achieved mAP = 0.995 , precison = 1 , Recall = 0.775 after 15 epochs training
 * Detected hand corectly in 12 out of 13 images

## GRAPHS
<img src="https://user-images.githubusercontent.com/66863370/154439843-9e821103-9f3f-4409-8f33-a3f21826fa92.png" width ="1720" height="720"/>

## Resulted inference video ( low quality)
<img src="https://user-images.githubusercontent.com/66863370/154436536-1af93aff-620f-4991-be46-8751728f8cb3.gif" width="720" height="640"/>

## Reference
https://github.com/cansik/yolo-hand-detection

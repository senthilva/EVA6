# YOLO

The goal here is to understand YOLO and implement for custom dataset

## Team mates

All dropped off. So for now - just me :)

## Object Detection

https://github.com/senthilva/EVA6/blob/main/session11/opencv_yolo.ipynb

![image](https://user-images.githubusercontent.com/8141261/127738702-dea63993-9a1f-4ba8-8b80-7893609b8b7d.png)


## Yolov3

Modified the below github 

https://github.com/senthilva/YoloV3-1

- Moved the data to drive and mounted from there 

```
  - filters=27
  - classes=4
  - burn_in to 100
  - max_batches to 5000
  - steps to 4000,4500
```

custom.names

```
  hardhat
  vest
  mask
  boots

```
custom.data


```
  classes=4
  train=data/customdata/train.txt
  valid=data/customdata/test.txt
  names=data/customdata/custom.names

```

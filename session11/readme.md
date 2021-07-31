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

- **Moved the data to drive and mounted from there** 

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

# Run the model on youtube video and test samples

## Output at 3 epochs

https://www.youtube.com/watch?v=X9Os8B6OglA

## Output at 30 epochs

https://www.youtube.com/watch?v=uoEhOC_elt8


## Output on few samples

## Boots
![image](https://user-images.githubusercontent.com/8141261/127745584-f311950f-b775-4da9-ac0e-a5e57d5df6e9.png) ![image](https://user-images.githubusercontent.com/8141261/127745587-6498f575-58d8-4d40-abd1-2101a094e2cb.png) ![image](https://user-images.githubusercontent.com/8141261/127745591-1ccfde7d-6b92-4997-a93c-a4fad086b209.png) ![image](https://user-images.githubusercontent.com/8141261/127745598-99ffed1c-b1bb-4e0f-959a-860fdbf0c184.png)

## Mask
![image](https://user-images.githubusercontent.com/8141261/127745611-95ca18e5-6996-444b-b15f-293d175d2df0.png) ![image](https://user-images.githubusercontent.com/8141261/127745619-210f7a80-7ee6-4c56-82c3-bcce715b71ee.png) ![image](https://user-images.githubusercontent.com/8141261/127745623-6571c36d-1f3d-4e1b-b5b0-7477315470a2.png) ![image](https://user-images.githubusercontent.com/8141261/127745628-a6efb149-63ce-41cf-b7c5-fe4a2979b8b7.png)

## Hard Hat

![image](https://user-images.githubusercontent.com/8141261/127745654-57a670ac-d397-4bc0-836f-c2c721c74582.png) ![image](https://user-images.githubusercontent.com/8141261/127745659-a63c190d-2b3e-4f2b-a3b0-aae195ce364c.png) ![image](https://user-images.githubusercontent.com/8141261/127745663-3d51f807-60a3-47a1-8369-285b8445bc09.png) ![image](https://user-images.githubusercontent.com/8141261/127745668-128a2135-9604-4534-81f1-3417d6f3a69e.png)

## Vest

![image](https://user-images.githubusercontent.com/8141261/127745674-a55e6907-097f-41e7-806f-f924cb45c747.png) ![image](https://user-images.githubusercontent.com/8141261/127745679-b21f44bc-e2c3-45cd-be99-a87dab897f13.png) ![image](https://user-images.githubusercontent.com/8141261/127745686-bb96df26-364b-43c5-87c9-9093d745639c.png) ![image](https://user-images.githubusercontent.com/8141261/127745687-c06407f9-71c6-45b0-bef3-8739963ce16c.png)





 






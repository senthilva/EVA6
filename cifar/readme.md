# Cifar

The goal here is to achieve 85% accuracy on the cifar 10 dataset using
- Depthwise Separable Convolution
- Dilated Convolution
- Not using Max Pooling
- Augmentation
  - horizontal flip   
  - shiftScaleRotate
  - coarseDropout



## Network Used

Below is the network we have used as a baseline.

```
  ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
            Conv2d-2           [-1, 32, 32, 32]           4,608
       BatchNorm2d-3           [-1, 32, 32, 32]              64
              ReLU-4           [-1, 32, 32, 32]               0
           Dropout-5           [-1, 32, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]              96
            Conv2d-7           [-1, 64, 15, 15]          18,432
            Conv2d-8          [-1, 128, 15, 15]          73,728
       BatchNorm2d-9          [-1, 128, 15, 15]             256
             ReLU-10          [-1, 128, 15, 15]               0
          Dropout-11          [-1, 128, 15, 15]               0
           Conv2d-12           [-1, 64, 11, 11]          73,728
           Conv2d-13             [-1, 32, 9, 9]          18,432
      BatchNorm2d-14             [-1, 32, 9, 9]              64
             ReLU-15             [-1, 32, 9, 9]               0
          Dropout-16             [-1, 32, 9, 9]               0
           Conv2d-17             [-1, 32, 9, 9]             320
           Conv2d-18             [-1, 16, 9, 9]             528
      BatchNorm2d-19             [-1, 16, 9, 9]              32
             ReLU-20             [-1, 16, 9, 9]               0
          Dropout-21             [-1, 16, 9, 9]               0
        AvgPool2d-22             [-1, 16, 1, 1]               0
           Conv2d-23             [-1, 10, 1, 1]             160
================================================================
Total params: 190,880
Trainable params: 190,880
Non-trainable params: 0
----------------------------------------------------------------

```

## Using albumentation

Built a custom transform class and function to be used with data load during transformation

```
  class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
```

```
  transform = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=12, max_width=12, min_holes = 1, min_height=1, min_width=1, fill_value=0.5, mask_fill_value = None),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
```

## Dilated Convolutions
```
  # Dilated convolutions
              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, bias=False,dilation=2),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
```
## Depthwise Seperable convolution
```
  # Depthwise Seperable convolution
              nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
              nn.Conv2d(32, 16, kernel_size=1),
```

## Complete Model

```
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()

          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), bias=False,padding = 1),
              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False, padding = 1),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.Dropout(0.2)

          ) # output_size = 32

          #skip connections
          self.skip1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), bias=False),

          ) # output

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False,stride=2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=False,padding = 1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Dropout(0.1)
          ) # output_size = 30

           #skip connections
          self.skip2 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),

          ) # out

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              # Dilated convolutions
              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, bias=False,dilation=2),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(),      
              nn.Dropout(0.1)
          ) # output_size = 12

          #self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11


          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              # Depthwise Seperable convolution
              nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
              nn.Conv2d(32, 16, kernel_size=1),
              nn.BatchNorm2d(16),
              nn.ReLU(), 
              nn.Dropout(0.1)
          ) # output_size = 9

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=9)
          ) # output_size = 1

          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

          ) 


      def forward(self, x):
          xin = x
          x = self.convblock1(x) + self.skip1(xin)
          x = self.convblock2(x) 
          x = self.convblock3(x)
          x = self.convblock4(x) 
          x = self.gap(x)        
          x = self.convblock5(x)

          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1)
```

## Results:

![image](https://user-images.githubusercontent.com/8141261/122593465-56c55c00-d083-11eb-9c4f-752c90f52503.png)

## Analysis:

- Adding 2 3x3 convolutions within the conv block helped with accuracy. Tried with using just 1 3x3 convolution within a block test accuracy did not go further than 76%
- Also tried using a skip connection over the first convolution block
- Augmentation used helped with reducing the gap between train and test accuracy
- Total params: 190,880
- There is still scope to improve the accuracy of the model. The train accuracy has not reached 90%
- Max Train accuracy : 84.67%
- Max Test accuracy  : 85.20% (47th epoch)
- No of epochs : 50


## Logs

```
     0%|          | 0/391 [00:00<?, ?it/s]EPOCH: 1
Loss=1.3616243600845337 Batch_id=390 Accuracy=38.37: 100%|██████████| 391/391 [00:11<00:00, 35.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0120, Accuracy: 4589/10000 (45.89%)

EPOCH: 2
Loss=1.2903900146484375 Batch_id=390 Accuracy=54.07: 100%|██████████| 391/391 [00:11<00:00, 35.21it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0093, Accuracy: 5783/10000 (57.83%)

EPOCH: 3
Loss=0.8459411859512329 Batch_id=390 Accuracy=61.11: 100%|██████████| 391/391 [00:11<00:00, 35.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0088, Accuracy: 6167/10000 (61.67%)

EPOCH: 4
Loss=1.0582202672958374 Batch_id=390 Accuracy=64.56: 100%|██████████| 391/391 [00:11<00:00, 35.01it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0074, Accuracy: 6717/10000 (67.17%)

EPOCH: 5
Loss=0.9144412279129028 Batch_id=390 Accuracy=67.79: 100%|██████████| 391/391 [00:11<00:00, 35.08it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0065, Accuracy: 7132/10000 (71.32%)

EPOCH: 6
Loss=0.875543475151062 Batch_id=390 Accuracy=69.60: 100%|██████████| 391/391 [00:11<00:00, 34.89it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0068, Accuracy: 7086/10000 (70.86%)

EPOCH: 7
Loss=0.9087523221969604 Batch_id=390 Accuracy=71.05: 100%|██████████| 391/391 [00:11<00:00, 35.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0064, Accuracy: 7187/10000 (71.87%)

EPOCH: 8
Loss=0.7223751544952393 Batch_id=390 Accuracy=72.35: 100%|██████████| 391/391 [00:11<00:00, 35.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0060, Accuracy: 7426/10000 (74.26%)

EPOCH: 9
Loss=0.6854380965232849 Batch_id=390 Accuracy=73.19: 100%|██████████| 391/391 [00:11<00:00, 34.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0055, Accuracy: 7604/10000 (76.04%)

EPOCH: 10
Loss=0.8621746897697449 Batch_id=390 Accuracy=74.53: 100%|██████████| 391/391 [00:11<00:00, 34.23it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0053, Accuracy: 7679/10000 (76.79%)

EPOCH: 11
Loss=0.5989867448806763 Batch_id=390 Accuracy=74.71: 100%|██████████| 391/391 [00:11<00:00, 35.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0051, Accuracy: 7772/10000 (77.72%)

EPOCH: 12
Loss=0.5655485391616821 Batch_id=390 Accuracy=76.12: 100%|██████████| 391/391 [00:11<00:00, 35.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0055, Accuracy: 7666/10000 (76.66%)

EPOCH: 13
Loss=0.7895005941390991 Batch_id=390 Accuracy=76.18: 100%|██████████| 391/391 [00:11<00:00, 35.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0051, Accuracy: 7840/10000 (78.40%)

EPOCH: 14
Loss=0.5183305144309998 Batch_id=390 Accuracy=77.13: 100%|██████████| 391/391 [00:11<00:00, 35.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0048, Accuracy: 7899/10000 (78.99%)

EPOCH: 15
Loss=0.5472713112831116 Batch_id=390 Accuracy=77.48: 100%|██████████| 391/391 [00:11<00:00, 35.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 7812/10000 (78.12%)

EPOCH: 16
Loss=0.408999502658844 Batch_id=390 Accuracy=77.81: 100%|██████████| 391/391 [00:10<00:00, 35.56it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0054, Accuracy: 7760/10000 (77.60%)

EPOCH: 17
Loss=0.9425004124641418 Batch_id=390 Accuracy=78.56: 100%|██████████| 391/391 [00:11<00:00, 35.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0050, Accuracy: 7852/10000 (78.52%)

EPOCH: 18
Loss=0.7217897176742554 Batch_id=390 Accuracy=78.64: 100%|██████████| 391/391 [00:11<00:00, 35.29it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0049, Accuracy: 7953/10000 (79.53%)

EPOCH: 19
Loss=0.6526190042495728 Batch_id=390 Accuracy=79.04: 100%|██████████| 391/391 [00:11<00:00, 35.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0048, Accuracy: 7927/10000 (79.27%)

EPOCH: 20
Loss=0.4442247748374939 Batch_id=390 Accuracy=79.47: 100%|██████████| 391/391 [00:11<00:00, 35.48it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0051, Accuracy: 7847/10000 (78.47%)

EPOCH: 21
Loss=0.4218319356441498 Batch_id=390 Accuracy=79.86: 100%|██████████| 391/391 [00:11<00:00, 35.39it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 8072/10000 (80.72%)

EPOCH: 22
Loss=0.6574229001998901 Batch_id=390 Accuracy=79.90: 100%|██████████| 391/391 [00:11<00:00, 35.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8182/10000 (81.82%)

EPOCH: 23
Loss=0.597236692905426 Batch_id=390 Accuracy=80.09: 100%|██████████| 391/391 [00:10<00:00, 35.78it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8138/10000 (81.38%)

EPOCH: 24
Loss=0.576559841632843 Batch_id=390 Accuracy=80.61: 100%|██████████| 391/391 [00:10<00:00, 35.87it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8223/10000 (82.23%)

EPOCH: 25
Loss=0.5489528775215149 Batch_id=390 Accuracy=80.48: 100%|██████████| 391/391 [00:11<00:00, 35.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0045, Accuracy: 8057/10000 (80.57%)

EPOCH: 26
Loss=0.6187418699264526 Batch_id=390 Accuracy=81.09: 100%|██████████| 391/391 [00:11<00:00, 35.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0047, Accuracy: 8041/10000 (80.41%)

EPOCH: 27
Loss=0.5735852122306824 Batch_id=390 Accuracy=81.39: 100%|██████████| 391/391 [00:10<00:00, 35.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8164/10000 (81.64%)

EPOCH: 28
Loss=0.5412876605987549 Batch_id=390 Accuracy=81.43: 100%|██████████| 391/391 [00:11<00:00, 35.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8212/10000 (82.12%)

EPOCH: 29
Loss=0.454658567905426 Batch_id=390 Accuracy=81.75: 100%|██████████| 391/391 [00:11<00:00, 35.36it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8253/10000 (82.53%)

EPOCH: 30
Loss=0.6937731504440308 Batch_id=390 Accuracy=81.86: 100%|██████████| 391/391 [00:11<00:00, 35.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8294/10000 (82.94%)

EPOCH: 31
Loss=0.5531013607978821 Batch_id=390 Accuracy=82.04: 100%|██████████| 391/391 [00:11<00:00, 35.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8328/10000 (83.28%)

EPOCH: 32
Loss=0.5086328983306885 Batch_id=390 Accuracy=82.22: 100%|██████████| 391/391 [00:10<00:00, 35.60it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8340/10000 (83.40%)

EPOCH: 33
Loss=0.4308083653450012 Batch_id=390 Accuracy=82.35: 100%|██████████| 391/391 [00:11<00:00, 34.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8365/10000 (83.65%)

EPOCH: 34
Loss=0.5718369483947754 Batch_id=390 Accuracy=82.58: 100%|██████████| 391/391 [00:11<00:00, 35.07it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8423/10000 (84.23%)

EPOCH: 35
Loss=0.411592960357666 Batch_id=390 Accuracy=82.73: 100%|██████████| 391/391 [00:11<00:00, 35.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8429/10000 (84.29%)

EPOCH: 36
Loss=0.5408097505569458 Batch_id=390 Accuracy=82.90: 100%|██████████| 391/391 [00:11<00:00, 35.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0039, Accuracy: 8387/10000 (83.87%)

EPOCH: 37
Loss=0.47667431831359863 Batch_id=390 Accuracy=83.01: 100%|██████████| 391/391 [00:11<00:00, 35.31it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8411/10000 (84.11%)

EPOCH: 38
Loss=0.5939132571220398 Batch_id=390 Accuracy=83.20: 100%|██████████| 391/391 [00:11<00:00, 35.41it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0039, Accuracy: 8383/10000 (83.83%)

EPOCH: 39
Loss=0.5559241771697998 Batch_id=390 Accuracy=83.17: 100%|██████████| 391/391 [00:11<00:00, 35.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8477/10000 (84.77%)

EPOCH: 40
Loss=0.4638153612613678 Batch_id=390 Accuracy=83.31: 100%|██████████| 391/391 [00:11<00:00, 34.43it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8474/10000 (84.74%)

EPOCH: 41
Loss=0.4561111330986023 Batch_id=390 Accuracy=83.57: 100%|██████████| 391/391 [00:11<00:00, 35.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8332/10000 (83.32%)

EPOCH: 42
Loss=0.3844226002693176 Batch_id=390 Accuracy=83.64: 100%|██████████| 391/391 [00:11<00:00, 34.98it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8411/10000 (84.11%)

EPOCH: 43
Loss=0.4064803719520569 Batch_id=390 Accuracy=83.81: 100%|██████████| 391/391 [00:11<00:00, 35.22it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0040, Accuracy: 8334/10000 (83.34%)

EPOCH: 44
Loss=0.3490757346153259 Batch_id=390 Accuracy=83.79: 100%|██████████| 391/391 [00:11<00:00, 35.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8406/10000 (84.06%)

EPOCH: 45
Loss=0.4174632430076599 Batch_id=390 Accuracy=84.22: 100%|██████████| 391/391 [00:11<00:00, 34.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8445/10000 (84.45%)

EPOCH: 46
Loss=0.47883373498916626 Batch_id=390 Accuracy=83.93: 100%|██████████| 391/391 [00:11<00:00, 35.05it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8484/10000 (84.84%)

EPOCH: 47
Loss=0.34756025671958923 Batch_id=390 Accuracy=84.21: 100%|██████████| 391/391 [00:11<00:00, 35.24it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8520/10000 (85.20%)

EPOCH: 48
Loss=0.39172428846359253 Batch_id=390 Accuracy=84.34: 100%|██████████| 391/391 [00:11<00:00, 35.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8439/10000 (84.39%)

EPOCH: 49
Loss=0.49151960015296936 Batch_id=390 Accuracy=84.19: 100%|██████████| 391/391 [00:11<00:00, 35.08it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8399/10000 (83.99%)

EPOCH: 50
Loss=0.4402956962585449 Batch_id=390 Accuracy=84.67: 100%|██████████| 391/391 [00:11<00:00, 34.63it/s]

Test set: Average loss: 0.0037, Accuracy: 8422/10000 (84.22%)


```



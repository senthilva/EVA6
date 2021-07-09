# Super convergence using 1 Cycle LR for CIFAR 10

The goal here is to write a custom ResNet architecture for CIFAR10 to achive 90+ accuracy in less than 24 epochs using 1 cycle LR strategy

## Library used

https://github.com/senthilva/deeplearning_template


## Custom Network built

https://github.com/senthilva/deeplearning_template/blob/main/models/custom_resnet.py

```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class custom_ResNet(nn.Module):
        def __init__(self,num_classes=10):
            super(custom_ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = nn.Sequential(
                    nn.Conv2d(self.in_planes, 128,
                              kernel_size=3, stride=1, padding=1,bias=False),
                    nn.MaxPool2d(2, stride=2),
                    nn.BatchNorm2d(128)
                )
            self.resblk1 = nn.Sequential(
                    nn.Conv2d(128, 128,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )
            self.layer2 = nn.Sequential(
                    nn.Conv2d(128, 256,
                              kernel_size=3, stride=1, padding=1,bias=False),
                    nn.MaxPool2d(2, stride=2),
                    nn.BatchNorm2d(256)
                )
            self.layer3 = nn.Sequential(
                    nn.Conv2d(256, 512,
                              kernel_size=3, stride=1, padding=1,bias=False),
                    nn.MaxPool2d(2, stride=2),
                    nn.BatchNorm2d(512)
                )
            self.resblk2 = nn.Sequential(
                    nn.Conv2d(512, 512,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 512,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(512),
                   nn.ReLU(),
                )
            self.linear = nn.Linear(512, num_classes)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x))) # prep layer
            X = F.relu(self.layer1(out))
            R1 = self.resblk1(X)
            out = R1 + X
            out = self.layer2(out)
            X = self.layer3(out)
            R2 = self.resblk2(X)
            out = R2 + X
            out = F.max_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = F.softmax(self.linear(out),dim = 1)
            return out


    def cust_ResNet18():
        return custom_ResNet()

```

## Image Augmentation used



    ```
     def transform_trainv2():
        return A.Compose(
        [
            A.CropAndPad(4,pad_mode=0, pad_cval=0, pad_cval_mask=0, 
                         keep_size=False, sample_independently=True, interpolation=1, 
                         always_apply=True),
            A.RandomCrop(32,32,always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=1, min_width=1, fill_value=0.5, mask_fill_value = None),
            A.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ])

    def transform_testv2():
        return A.Compose(
        [
            A.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ])
    ```


## 1 Cycle LR used

    ```

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4,
                                                    pct_start = 0.18,
                                                    steps_per_epoch=len(trainloader),
                                                    epochs=26)
    ```


## Results:



## Analysis:

- After building the network , it took a couple of iterations to find the max LR and min LR
    - Tried with max LR of 1 and 2. Max test accuracy was at 24 epoch of ~85% at 24 epochs
    - Reduced max LR to 0.4 Max test accuracy was 88.59% at 24 spochs





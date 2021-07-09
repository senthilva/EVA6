# Super convergence using 1 Cycle LR for CIFAR 10

The goal here is to write a custom ResNet architecture for CIFAR10 to achive 90+ accuracy in less than 24 epochs using 1 cycle LR strategy

## Team mates

Nishad, Prasad, Owais, Senthil

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
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
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

   ```
   
    Epoch: 1
    /usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
     [================================================================>]  Step: 1s271ms | Tot: 14s515ms | Loss: 2.119 | Acc: 33.636% (16818/50000) 98/98 
     [=============================================================>...]  Step: 393ms | Tot: 1s191ms | Loss: 2.074 | Acc: 38.090% (3809/10000) 20/20 
    Saving..
     Learning Rate : [0.05865921643637467]

    Epoch: 2
     [================================================================>]  Step: 88ms | Tot: 13s477ms | Loss: 2.033 | Acc: 42.238% (21119/50000) 98/98 
     [=============================================================>...]  Step: 23ms | Tot: 822ms | Loss: 2.030 | Acc: 42.670% (4267/10000) 20/20 
    Saving..
     Learning Rate : [0.1676805246312752]

    Epoch: 3
     [================================================================>]  Step: 90ms | Tot: 13s434ms | Loss: 2.027 | Acc: 42.936% (21468/50000) 98/98 
     [=============================================================>...]  Step: 23ms | Tot: 830ms | Loss: 2.059 | Acc: 40.100% (4010/10000) 20/20 
     Learning Rate : [0.2946184706007123]

    Epoch: 4
     [================================================================>]  Step: 91ms | Tot: 13s663ms | Loss: 2.011 | Acc: 44.626% (22313/50000) 98/98 
     [=============================================================>...]  Step: 24ms | Tot: 847ms | Loss: 1.991 | Acc: 46.730% (4673/10000) 20/20 
    Saving..
     Learning Rate : [0.38306604068740413]

    Epoch: 5
     [================================================================>]  Step: 93ms | Tot: 13s801ms | Loss: 1.963 | Acc: 49.442% (24721/50000) 98/98 
     [=============================================================>...]  Step: 25ms | Tot: 846ms | Loss: 1.939 | Acc: 51.790% (5179/10000) 20/20 
    Saving..
     Learning Rate : [0.399684520617609]

    Epoch: 6
     [================================================================>]  Step: 95ms | Tot: 14s8ms | Loss: 1.924 | Acc: 53.350% (26675/50000) 98/98 
     [=============================================================>...]  Step: 31ms | Tot: 872ms | Loss: 1.930 | Acc: 52.830% (5283/10000) 20/20 
    Saving..
     Learning Rate : [0.39581087341411936]

    Epoch: 7
     [================================================================>]  Step: 96ms | Tot: 14s269ms | Loss: 1.883 | Acc: 57.590% (28795/50000) 98/98 
     [=============================================================>...]  Step: 32ms | Tot: 902ms | Loss: 1.868 | Acc: 59.350% (5935/10000) 20/20 
    Saving..
     Learning Rate : [0.3876063425675938]

    Epoch: 8
     [================================================================>]  Step: 99ms | Tot: 14s829ms | Loss: 1.811 | Acc: 64.984% (32492/50000) 98/98 
     [=============================================================>...]  Step: 28ms | Tot: 936ms | Loss: 1.806 | Acc: 65.470% (6547/10000) 20/20 
    Saving..
     Learning Rate : [0.37525239406608124]

    Epoch: 9
     [================================================================>]  Step: 100ms | Tot: 14s579ms | Loss: 1.782 | Acc: 67.964% (33982/50000) 98/98 
     [=============================================================>...]  Step: 28ms | Tot: 901ms | Loss: 1.818 | Acc: 64.440% (6444/10000) 20/20 
     Learning Rate : [0.35902226979422636]

    Epoch: 10
     [================================================================>]  Step: 96ms | Tot: 14s265ms | Loss: 1.739 | Acc: 72.766% (36383/50000) 98/98 
     [=============================================================>...]  Step: 28ms | Tot: 881ms | Loss: 1.766 | Acc: 69.720% (6972/10000) 20/20 
    Saving..
     Learning Rate : [0.33927494403005204]

    Epoch: 11
     [================================================================>]  Step: 97ms | Tot: 14s195ms | Loss: 1.714 | Acc: 75.444% (37722/50000) 98/98 
     [=============================================================>...]  Step: 30ms | Tot: 862ms | Loss: 1.780 | Acc: 68.190% (6819/10000) 20/20 
     Learning Rate : [0.31644718373173647]

    Epoch: 12
     [================================================================>]  Step: 96ms | Tot: 14s199ms | Loss: 1.701 | Acc: 76.626% (38313/50000) 98/98 
     [=============================================================>...]  Step: 31ms | Tot: 895ms | Loss: 1.706 | Acc: 76.020% (7602/10000) 20/20 
    Saving..
     Learning Rate : [0.29104388822319777]

    Epoch: 13
     [================================================================>]  Step: 97ms | Tot: 14s384ms | Loss: 1.684 | Acc: 78.474% (39237/50000) 98/98 
     [=============================================================>...]  Step: 29ms | Tot: 890ms | Loss: 1.700 | Acc: 77.030% (7703/10000) 20/20 
    Saving..
     Learning Rate : [0.26362692194316906]

    Epoch: 14
     [================================================================>]  Step: 96ms | Tot: 14s482ms | Loss: 1.672 | Acc: 79.676% (39838/50000) 98/98 
     [=============================================================>...]  Step: 29ms | Tot: 896ms | Loss: 1.691 | Acc: 78.170% (7817/10000) 20/20 
    Saving..
     Learning Rate : [0.23480268725253775]

    Epoch: 15
     [================================================================>]  Step: 95ms | Tot: 14s345ms | Loss: 1.666 | Acc: 80.536% (40268/50000) 98/98 
     [=============================================================>...]  Step: 32ms | Tot: 876ms | Loss: 1.677 | Acc: 79.290% (7929/10000) 20/20 
    Saving..
     Learning Rate : [0.20520871216182993]

    Epoch: 16
     [================================================================>]  Step: 98ms | Tot: 14s319ms | Loss: 1.654 | Acc: 81.662% (40831/50000) 98/98 
     [=============================================================>...]  Step: 30ms | Tot: 885ms | Loss: 1.664 | Acc: 80.860% (8086/10000) 20/20 
    Saving..
     Learning Rate : [0.17549954962850084]

    Epoch: 17
     [================================================================>]  Step: 95ms | Tot: 14s255ms | Loss: 1.644 | Acc: 82.696% (41348/50000) 98/98 
     [=============================================================>...]  Step: 34ms | Tot: 877ms | Loss: 1.679 | Acc: 78.940% (7894/10000) 20/20 
     Learning Rate : [0.1463323003002368]

    Epoch: 18
     [================================================================>]  Step: 97ms | Tot: 14s320ms | Loss: 1.635 | Acc: 83.670% (41835/50000) 98/98 
     [=============================================================>...]  Step: 25ms | Tot: 937ms | Loss: 1.652 | Acc: 81.710% (8171/10000) 20/20 
    Saving..
     Learning Rate : [0.1183520789090106]

    Epoch: 19
     [================================================================>]  Step: 98ms | Tot: 14s401ms | Loss: 1.625 | Acc: 84.734% (42367/50000) 98/98 
     [=============================================================>...]  Step: 26ms | Tot: 964ms | Loss: 1.662 | Acc: 80.700% (8070/10000) 20/20 
     Learning Rate : [0.0921777457669548]

    Epoch: 20
     [================================================================>]  Step: 96ms | Tot: 14s328ms | Loss: 1.616 | Acc: 85.568% (42784/50000) 98/98 
     [=============================================================>...]  Step: 26ms | Tot: 967ms | Loss: 1.636 | Acc: 83.270% (8327/10000) 20/20 
    Saving..
     Learning Rate : [0.06838821895165247]

    Epoch: 21
     [================================================================>]  Step: 96ms | Tot: 14s414ms | Loss: 1.603 | Acc: 86.728% (43364/50000) 98/98 
     [=============================================================>...]  Step: 26ms | Tot: 978ms | Loss: 1.617 | Acc: 85.220% (8522/10000) 20/20 
    Saving..
     Learning Rate : [0.04750966992488575]

    Epoch: 22
     [================================================================>]  Step: 98ms | Tot: 14s486ms | Loss: 1.592 | Acc: 87.914% (43957/50000) 98/98 
     [=============================================================>...]  Step: 25ms | Tot: 935ms | Loss: 1.607 | Acc: 86.420% (8642/10000) 20/20 
    Saving..
     Learning Rate : [0.0300038857892982]

    Epoch: 23
     [================================================================>]  Step: 96ms | Tot: 14s383ms | Loss: 1.578 | Acc: 89.338% (44669/50000) 98/98 
     [=============================================================>...]  Step: 25ms | Tot: 953ms | Loss: 1.589 | Acc: 88.060% (8806/10000) 20/20 
    Saving..
     Learning Rate : [0.016258055583989996]

    Epoch: 24
     [================================================================>]  Step: 96ms | Tot: 14s359ms | Loss: 1.570 | Acc: 90.134% (45067/50000) 98/98 
     [=============================================================>...]  Step: 30ms | Tot: 889ms | Loss: 1.583 | Acc: 88.590% (8859/10000) 20/20 
    Saving..
     Learning Rate : [0.006576206523486491]


   ```

## Analysis:

- After building the network , it took a couple of iterations to find the max LR and min LR
    - Tried with max LR of 1 and 2. Max test accuracy was at 24 epoch of ~85% at 24 epochs
    - Reduced max LR to 0.4 Max test accuracy was 88.59% at 24 spochs





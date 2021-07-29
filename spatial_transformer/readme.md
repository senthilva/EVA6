
# Spatial Transformer for CIFAR10

The goal here is to implement the spatial transformer for CIFAR10



## Reference 

https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html



## Changes made

## Loading data

```
    batch_size = 64
    num_workers = 4

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

```

## Network

    Changes done to incorporate 3 channels and modify the FC to accomodate 32x32 vs 28x28

    ```
     class Net(nn.Module):
          def __init__(self):
              super(Net, self).__init__()
              self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
              self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
              self.conv2_drop = nn.Dropout2d()
              self.fc1 = nn.Linear(500, 50)
              self.fc2 = nn.Linear(50, 10)

              # Spatial transformer localization-network
              self.localization = nn.Sequential(
                  nn.Conv2d(3, 8, kernel_size=7),
                  nn.MaxPool2d(2, stride=2),
                  nn.ReLU(True),
                  nn.Conv2d(8, 10, kernel_size=5),
                  nn.MaxPool2d(2, stride=2),
                  nn.ReLU(True)
              )

              # Regressor for the 3 * 2 affine matrix
              self.fc_loc = nn.Sequential(
                  nn.Linear(10 * 4 * 4, 32),
                  nn.ReLU(True),
                  nn.Linear(32, 3 * 2)
              )

              # Initialize the weights/bias with identity transformation
              self.fc_loc[2].weight.data.zero_()
              self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

          # Spatial transformer network forward function
          def stn(self, x):
              xs = self.localization(x)
              #print(f'xs size : {xs.size()}')
              xs = xs.view(-1, 10 * 4*4)
              theta = self.fc_loc(xs)
              #print(f'x size : {x.size()}')
              theta = theta.view(-1, 2, 3)
              #print(f'theta size : {theta.size()}')
              grid = F.affine_grid(theta, x.size())
              x = F.grid_sample(x, grid)

              return x

          def forward(self, x):
              # transform the input
              x = self.stn(x)

              # Perform the usual forward pass
              x = F.relu(F.max_pool2d(self.conv1(x), 2))
              x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
              #print(x.size())
              x = x.view(-1, 500)
              #print(x.size())
              x = F.relu(self.fc1(x))
              x = F.dropout(x, training=self.training)
              x = self.fc2(x)
              return F.log_softmax(x, dim=1)


     model = Net().to(device)
          ```

## Results:

   ```
   
    Train Epoch: 1 [0/50000 (0%)]	Loss: 2.283560
    Train Epoch: 1 [32000/50000 (64%)]	Loss: 2.213211

    Test set: Average loss: 2.1066, Accuracy: 2662/10000 (27%)

    Train Epoch: 2 [0/50000 (0%)]	Loss: 2.243742
    Train Epoch: 2 [32000/50000 (64%)]	Loss: 2.083368

    Test set: Average loss: 1.9186, Accuracy: 3292/10000 (33%)

    Train Epoch: 3 [0/50000 (0%)]	Loss: 2.011009
    Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.870690

    Test set: Average loss: 1.8133, Accuracy: 3698/10000 (37%)

    Train Epoch: 4 [0/50000 (0%)]	Loss: 1.881079
    Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.893529

    Test set: Average loss: 1.7157, Accuracy: 3936/10000 (39%)

    Train Epoch: 5 [0/50000 (0%)]	Loss: 1.722593
    Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.820404

    Test set: Average loss: 1.6363, Accuracy: 4068/10000 (41%)

    Train Epoch: 6 [0/50000 (0%)]	Loss: 1.775517
    Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.794123

    Test set: Average loss: 1.6019, Accuracy: 4265/10000 (43%)

    Train Epoch: 7 [0/50000 (0%)]	Loss: 1.785279
    Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.690611

    Test set: Average loss: 1.5636, Accuracy: 4326/10000 (43%)

    Train Epoch: 8 [0/50000 (0%)]	Loss: 1.571787
    Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.911530

    Test set: Average loss: 1.5438, Accuracy: 4477/10000 (45%)

    Train Epoch: 9 [0/50000 (0%)]	Loss: 1.649011
    Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.737918

    Test set: Average loss: 1.5220, Accuracy: 4540/10000 (45%)

    Train Epoch: 10 [0/50000 (0%)]	Loss: 1.793884
    Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.872865

    Test set: Average loss: 1.5169, Accuracy: 4582/10000 (46%)

    Train Epoch: 11 [0/50000 (0%)]	Loss: 1.544499
    Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.504514

    Test set: Average loss: 1.5228, Accuracy: 4621/10000 (46%)

    Train Epoch: 12 [0/50000 (0%)]	Loss: 1.665646
    Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.582601

    Test set: Average loss: 1.4685, Accuracy: 4673/10000 (47%)

    Train Epoch: 13 [0/50000 (0%)]	Loss: 1.599882
    Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.573556

    Test set: Average loss: 1.4516, Accuracy: 4770/10000 (48%)

    Train Epoch: 14 [0/50000 (0%)]	Loss: 1.845129
    Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.572244

    Test set: Average loss: 1.4487, Accuracy: 4800/10000 (48%)

    Train Epoch: 15 [0/50000 (0%)]	Loss: 1.630089
    Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.669301

    Test set: Average loss: 1.4365, Accuracy: 4841/10000 (48%)

    Train Epoch: 16 [0/50000 (0%)]	Loss: 1.593873
    Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.567650

    Test set: Average loss: 1.4162, Accuracy: 4948/10000 (49%)

    Train Epoch: 17 [0/50000 (0%)]	Loss: 1.540712
    Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.514471

    Test set: Average loss: 1.4089, Accuracy: 4927/10000 (49%)

    Train Epoch: 18 [0/50000 (0%)]	Loss: 1.541935
    Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.431668

    Test set: Average loss: 1.4034, Accuracy: 5050/10000 (50%)

    Train Epoch: 19 [0/50000 (0%)]	Loss: 1.579468
    Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.614561

    Test set: Average loss: 1.3928, Accuracy: 5025/10000 (50%)

    Train Epoch: 20 [0/50000 (0%)]	Loss: 1.729503
    Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.453787

    Test set: Average loss: 1.3662, Accuracy: 5024/10000 (50%)

    Train Epoch: 21 [0/50000 (0%)]	Loss: 1.325983
    Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.470959

    Test set: Average loss: 1.3800, Accuracy: 5060/10000 (51%)

    Train Epoch: 22 [0/50000 (0%)]	Loss: 1.564264
    Train Epoch: 22 [32000/50000 (64%)]	Loss: 1.487486

    Test set: Average loss: 1.3891, Accuracy: 5047/10000 (50%)

    Train Epoch: 23 [0/50000 (0%)]	Loss: 1.668020
    Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.603889

    Test set: Average loss: 1.3955, Accuracy: 5062/10000 (51%)

    Train Epoch: 24 [0/50000 (0%)]	Loss: 1.461158
    Train Epoch: 24 [32000/50000 (64%)]	Loss: 1.311243

    Test set: Average loss: 1.3468, Accuracy: 5304/10000 (53%)

    Train Epoch: 25 [0/50000 (0%)]	Loss: 1.583642
    Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.621246

    Test set: Average loss: 1.3262, Accuracy: 5321/10000 (53%)

    Train Epoch: 26 [0/50000 (0%)]	Loss: 1.597041
    Train Epoch: 26 [32000/50000 (64%)]	Loss: 1.613701

    Test set: Average loss: 1.3190, Accuracy: 5240/10000 (52%)

    Train Epoch: 27 [0/50000 (0%)]	Loss: 1.751663
    Train Epoch: 27 [32000/50000 (64%)]	Loss: 1.290218

    Test set: Average loss: 1.3467, Accuracy: 5265/10000 (53%)

    Train Epoch: 28 [0/50000 (0%)]	Loss: 1.675216
    Train Epoch: 28 [32000/50000 (64%)]	Loss: 1.596299

    Test set: Average loss: 1.2914, Accuracy: 5409/10000 (54%)

    Train Epoch: 29 [0/50000 (0%)]	Loss: 1.410681
    Train Epoch: 29 [32000/50000 (64%)]	Loss: 1.519847

    Test set: Average loss: 1.3010, Accuracy: 5380/10000 (54%)

    Train Epoch: 30 [0/50000 (0%)]	Loss: 1.456616
    Train Epoch: 30 [32000/50000 (64%)]	Loss: 1.436046

    Test set: Average loss: 1.2753, Accuracy: 5475/10000 (55%)

    Train Epoch: 31 [0/50000 (0%)]	Loss: 1.304710
    Train Epoch: 31 [32000/50000 (64%)]	Loss: 1.397923

    Test set: Average loss: 1.2596, Accuracy: 5568/10000 (56%)

    Train Epoch: 32 [0/50000 (0%)]	Loss: 1.509818
    Train Epoch: 32 [32000/50000 (64%)]	Loss: 1.477103

    Test set: Average loss: 1.2810, Accuracy: 5511/10000 (55%)

    Train Epoch: 33 [0/50000 (0%)]	Loss: 1.585972
    Train Epoch: 33 [32000/50000 (64%)]	Loss: 1.478205

    Test set: Average loss: 1.3600, Accuracy: 5243/10000 (52%)

    Train Epoch: 34 [0/50000 (0%)]	Loss: 1.711602
    Train Epoch: 34 [32000/50000 (64%)]	Loss: 1.443586

    Test set: Average loss: 1.2901, Accuracy: 5394/10000 (54%)

    Train Epoch: 35 [0/50000 (0%)]	Loss: 1.795575
    Train Epoch: 35 [32000/50000 (64%)]	Loss: 1.331098

    Test set: Average loss: 1.2625, Accuracy: 5525/10000 (55%)

    Train Epoch: 36 [0/50000 (0%)]	Loss: 1.498483
    Train Epoch: 36 [32000/50000 (64%)]	Loss: 1.315819

    Test set: Average loss: 1.2984, Accuracy: 5419/10000 (54%)

    Train Epoch: 37 [0/50000 (0%)]	Loss: 1.556057
    Train Epoch: 37 [32000/50000 (64%)]	Loss: 1.239275

    Test set: Average loss: 1.2448, Accuracy: 5621/10000 (56%)

    Train Epoch: 38 [0/50000 (0%)]	Loss: 1.323399
    Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.278013

    Test set: Average loss: 1.2276, Accuracy: 5638/10000 (56%)

    Train Epoch: 39 [0/50000 (0%)]	Loss: 1.452869
    Train Epoch: 39 [32000/50000 (64%)]	Loss: 1.292807

    Test set: Average loss: 1.2577, Accuracy: 5551/10000 (56%)

    Train Epoch: 40 [0/50000 (0%)]	Loss: 1.443661
    Train Epoch: 40 [32000/50000 (64%)]	Loss: 1.417369

    Test set: Average loss: 1.2171, Accuracy: 5746/10000 (57%)

    Train Epoch: 41 [0/50000 (0%)]	Loss: 1.713758
    Train Epoch: 41 [32000/50000 (64%)]	Loss: 1.471299

    Test set: Average loss: 1.2123, Accuracy: 5739/10000 (57%)

    Train Epoch: 42 [0/50000 (0%)]	Loss: 1.624909
    Train Epoch: 42 [32000/50000 (64%)]	Loss: 1.512395

    Test set: Average loss: 1.2380, Accuracy: 5667/10000 (57%)

    Train Epoch: 43 [0/50000 (0%)]	Loss: 1.216347
    Train Epoch: 43 [32000/50000 (64%)]	Loss: 1.284247

    Test set: Average loss: 1.2122, Accuracy: 5770/10000 (58%)

    Train Epoch: 44 [0/50000 (0%)]	Loss: 1.310111
    Train Epoch: 44 [32000/50000 (64%)]	Loss: 1.348832

    Test set: Average loss: 1.2134, Accuracy: 5797/10000 (58%)

    Train Epoch: 45 [0/50000 (0%)]	Loss: 1.424465
    Train Epoch: 45 [32000/50000 (64%)]	Loss: 1.409637

    Test set: Average loss: 1.2193, Accuracy: 5714/10000 (57%)

    Train Epoch: 46 [0/50000 (0%)]	Loss: 1.316475
    Train Epoch: 46 [32000/50000 (64%)]	Loss: 1.217960

    Test set: Average loss: 1.2435, Accuracy: 5561/10000 (56%)

    Train Epoch: 47 [0/50000 (0%)]	Loss: 1.499499
    Train Epoch: 47 [32000/50000 (64%)]	Loss: 1.209549

    Test set: Average loss: 1.1974, Accuracy: 5767/10000 (58%)

    Train Epoch: 48 [0/50000 (0%)]	Loss: 1.357558
    Train Epoch: 48 [32000/50000 (64%)]	Loss: 1.354312

    Test set: Average loss: 1.1835, Accuracy: 5840/10000 (58%)

    Train Epoch: 49 [0/50000 (0%)]	Loss: 1.420272
    Train Epoch: 49 [32000/50000 (64%)]	Loss: 1.380257

    Test set: Average loss: 1.2169, Accuracy: 5739/10000 (57%)

    Train Epoch: 50 [0/50000 (0%)]	Loss: 1.487736
    Train Epoch: 50 [32000/50000 (64%)]	Loss: 1.306191

    Test set: Average loss: 1.2250, Accuracy: 5693/10000 (57%)


   ```


![image](https://user-images.githubusercontent.com/8141261/127464872-bfb3b3fe-c7c4-4492-9c64-dbbec2f3cb7c.png)




## Analysis:

- We see that the network learns spatial transformations on the input images





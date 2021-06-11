# Batch Normalization

The goal here is to understand various batch normalization techniques on a neural network using the mnist dataset.
- Network with Group Normalization
- Network with Layer Normalization
- Network with L1 + BN

## Network Used

Below is the network we have used as a baseline.

```
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1           [-1, 10, 26, 26]              90
                ReLU-2           [-1, 10, 26, 26]               0
           GroupNorm-3           [-1, 10, 26, 26]              20
             Dropout-4           [-1, 10, 26, 26]               0
              Conv2d-5           [-1, 20, 24, 24]           1,800
                ReLU-6           [-1, 20, 24, 24]               0
           GroupNorm-7           [-1, 20, 24, 24]              40
              Conv2d-8           [-1, 10, 24, 24]             200
           MaxPool2d-9           [-1, 10, 12, 12]               0
             Conv2d-10           [-1, 20, 10, 10]           1,800
               ReLU-11           [-1, 20, 10, 10]               0
          GroupNorm-12           [-1, 20, 10, 10]              40
            Dropout-13           [-1, 20, 10, 10]               0
             Conv2d-14             [-1, 12, 8, 8]           2,160
               ReLU-15             [-1, 12, 8, 8]               0
          GroupNorm-16             [-1, 12, 8, 8]              24
             Conv2d-17             [-1, 12, 6, 6]           1,296
               ReLU-18             [-1, 12, 6, 6]               0
          GroupNorm-19             [-1, 12, 6, 6]              24
            Dropout-20             [-1, 12, 6, 6]               0
          AvgPool2d-21             [-1, 12, 1, 1]               0
             Conv2d-22             [-1, 10, 1, 1]             120
  ================================================================

```

## Batch Normalization Used

Defined  a batch normalization function that takes in batch normalization type (bn_type) and channels.

- Group Normalization - splitting into 2 groups
- Layer Normalization - Using GroupNorm function but using one group
- Batch Normalization - Using the usual nn.BatchNorm2d

```
  def batch_norm(bn_type,channels):
    if bn_type == 'GN':
      return nn.GroupNorm(2,channels)
    elif bn_type == 'LN':
      return nn.GroupNorm(1,channels)
    elif bn_type == 'BN':
      return nn.BatchNorm2d(channels)
```

Modified the class to take 2 parameters batch normalization type(bn_type) and dropout(dropout_value).

```
  class Net(nn.Module):
      def __init__(self,bn_type,dropout_value):
          super(Net, self).__init__()

          self.bn_type = bn_type
          self.dropout = dropout_value
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              batch_norm(self.bn_type,10),
              nn.Dropout(self.dropout)
          ) # output_size = 26

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              batch_norm(self.bn_type,20),
              #nn.Dropout(self.dropout)
          ) # output_size = 24

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 24
          self.pool1 = nn.MaxPool2d(2, 2) 

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),            
              batch_norm(self.bn_type,20),
              nn.Dropout(self.dropout)
          ) # output_size = 10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),            
              batch_norm(self.bn_type,12),
              #nn.Dropout(self.dropout)
          ) # output_size = 8
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),            
              batch_norm(self.bn_type,12),
              nn.Dropout(self.dropout)
          ) # output_size = 6
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),            
              batch_norm(self.bn_type,16),
              nn.Dropout(self.dropout)
          ) # output_size = 6

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=6)
          ) # output_size = 1

          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
              # nn.BatchNorm2d(10),
              # nn.ReLU(),
              # nn.Dropout(self.dropout)
          ) 


          self.dropout = nn.Dropout(dropout_value)

      def forward(self, x):
          x = self.convblock1(x)
          x = self.convblock2(x)
          x = self.convblock3(x)
          x = self.pool1(x)
          x = self.convblock4(x)
          x = self.convblock5(x)
          x = self.convblock6(x)
          #x = self.convblock7(x)
          x = self.gap(x)        
          x = self.convblock8(x)

          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1)
```


## Results:
![image](https://user-images.githubusercontent.com/8141261/121629176-4405c280-ca98-11eb-9ea2-a9a6cf348114.png)


## Analysis:


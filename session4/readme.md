
# MNIST

The objective is achieve a test accuracy > 99.4% with less than 18K parameters within 19 epochs.


## Steps taken

*  Reduce parameters : reduced the channels max 32 used
*  Reduce overfitting : Added batch normalization and dropout
*  Data augmentation
  *  Random rotation 5 degress
  *  Colour Jitter


# Network Design

* Convolution block having 2 convolution layers followed by batch normalization
* One dropout for each convolution block
* MaxPool after convolution block
* Global average pooling before last linear layer

  ```
    ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1            [-1, 8, 28, 28]              80
         BatchNorm2d-2            [-1, 8, 28, 28]              16
              Conv2d-3           [-1, 16, 28, 28]           1,168
         BatchNorm2d-4           [-1, 16, 28, 28]              32
             Dropout-5           [-1, 16, 28, 28]               0
           MaxPool2d-6           [-1, 16, 14, 14]               0
              Conv2d-7           [-1, 32, 14, 14]           4,640
         BatchNorm2d-8           [-1, 32, 14, 14]              64
              Conv2d-9           [-1, 24, 14, 14]           6,936
        BatchNorm2d-10           [-1, 24, 14, 14]              48
            Dropout-11           [-1, 24, 14, 14]               0
          MaxPool2d-12             [-1, 24, 7, 7]               0
             Conv2d-13             [-1, 16, 5, 5]           3,472
          AvgPool2d-14             [-1, 16, 1, 1]               0
             Linear-15                   [-1, 10]             170
  ================================================================
  Total params: 16,626
  Trainable params: 16,626
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.62
  Params size (MB): 0.06
  Estimated Total Size (MB): 0.69
  
  ```
 
## Training and Loss

* Number of epochs : 19
* Loss - Negative log likehood
* batch_size = 128
* 

  ```
  Epoch: 1, loss: 8588.646249156445, Classification Acc: 72.58666666666667, Addition Acc: 52.27
  Epoch: 2, loss: 1329.7578395307064, Classification Acc: 96.93, Addition Acc: 91.17333333333333
  Epoch: 3, loss: 968.9839583390858, Classification Acc: 97.88, Addition Acc: 95.56333333333333
  Epoch: 4, loss: 720.8626747094095, Classification Acc: 98.36, Addition Acc: 96.21666666666667
  Epoch: 5, loss: 597.3828105715802, Classification Acc: 98.67833333333334, Addition Acc: 97.28666666666666
  Epoch: 6, loss: 506.67549971531844, Classification Acc: 98.86666666666667, Addition Acc: 98.33666666666666
  Epoch: 7, loss: 450.0150337165105, Classification Acc: 98.98833333333333, Addition Acc: 98.30499999999999
  Epoch: 8, loss: 370.68281011172803, Classification Acc: 99.17, Addition Acc: 98.87
  Epoch: 9, loss: 342.45800989316194, Classification Acc: 99.21833333333333, Addition Acc: 98.965
  Epoch: 10, loss: 302.226245637954, Classification Acc: 99.325, Addition Acc: 99.08500000000001
  Finished Training

  ```

## Observations/ Learning

* Total params: 16,626
* Test accuracy achieved : 
* No batch normalization or dropout was used to close to the prediction layer

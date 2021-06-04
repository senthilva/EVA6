# 5. Reduce number of parameters

## Target:
Reduce the number of parameters to < 8K and maintain acuracy
- Reduced the channel dimesnions
- Reduced convolutions whne size was at 6x6
- Removed dropouts after 2 convolution layers as it was underfitting

## Results:
- Parameters: 7614 parameters
- Best Train Accuracy: 99.05
- Best Test Accuracy: 99.46

## Analysis:
- Reducing dropouts helped with overcoming nunderfitting
- Last 4 epochs have an average accuracy of 99.40%.

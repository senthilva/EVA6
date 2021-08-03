
# Spatial Transformer for CIFAR10

The goal here is to implement the spatial transformer for CIFAR10



## Data Set

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

## Loading data

Loaded the data directly using the kaggle api
    ```
        !kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
    ```

## Network

Changes done to incorporate 3 channels and modify the FC to accomodate 32x32 vs 28x28

    ```
     model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            transformer=efficient_transformer,
            channels=3,
                ).to(device)

    ```

## Results:

   ```
   
    


   ```






## Analysis:







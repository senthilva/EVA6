
# Visual Transformer for the Cats vs Dogs Dataset

The goal here is to implement the visual transformer for cats vs dogs data set



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
       100%
    313/313 [13:12<00:00, 2.53s/it]

    Epoch : 1 - loss : 0.6955 - acc: 0.5093 - val_loss : 0.6908 - val_acc: 0.5435

    100%
    313/313 [10:33<00:00, 2.02s/it]

    Epoch : 2 - loss : 0.6920 - acc: 0.5237 - val_loss : 0.6854 - val_acc: 0.5580

    100%
    313/313 [02:38<00:00, 1.97it/s]

    Epoch : 3 - loss : 0.6838 - acc: 0.5557 - val_loss : 0.6749 - val_acc: 0.5860

    100%
    313/313 [05:16<00:00, 1.01s/it]

    Epoch : 4 - loss : 0.6747 - acc: 0.5781 - val_loss : 0.6661 - val_acc: 0.5995

    100%
    313/313 [02:39<00:00, 1.97it/s]

    Epoch : 5 - loss : 0.6711 - acc: 0.5813 - val_loss : 0.6662 - val_acc: 0.5874

    100%
    313/313 [02:11<00:00, 2.38it/s]

    Epoch : 6 - loss : 0.6577 - acc: 0.5993 - val_loss : 0.6448 - val_acc: 0.6191

    100%
    313/313 [05:15<00:00, 1.01s/it]

    Epoch : 7 - loss : 0.6492 - acc: 0.6117 - val_loss : 0.6401 - val_acc: 0.6266

    100%
    313/313 [02:37<00:00, 1.98it/s]

    Epoch : 8 - loss : 0.6400 - acc: 0.6201 - val_loss : 0.6319 - val_acc: 0.6388

    100%
    313/313 [02:09<00:00, 2.42it/s]

    Epoch : 9 - loss : 0.6352 - acc: 0.6229 - val_loss : 0.6453 - val_acc: 0.6161

    100%
    313/313 [05:11<00:00, 1.00it/s]

    Epoch : 10 - loss : 0.6247 - acc: 0.6388 - val_loss : 0.6159 - val_acc: 0.6632

    100%
    313/313 [02:35<00:00, 2.01it/s]

    Epoch : 11 - loss : 0.6225 - acc: 0.6439 - val_loss : 0.6063 - val_acc: 0.6626

    100%
    313/313 [02:10<00:00, 2.40it/s]

    Epoch : 12 - loss : 0.6123 - acc: 0.6551 - val_loss : 0.6212 - val_acc: 0.6475

    100%
    313/313 [02:08<00:00, 2.43it/s]

    Epoch : 13 - loss : 0.6054 - acc: 0.6636 - val_loss : 0.6012 - val_acc: 0.6786

    69%
    216/313 [01:30<00:38, 2.50it/s]

    Epoch : 14 - loss : 0.6011 - acc: 0.6656 - val_loss : 0.5939 - val_acc: 0.6707

    HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

    Epoch : 15 - loss : 0.5976 - acc: 0.6728 - val_loss : 0.5924 - val_acc: 0.6764

    HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

    Epoch : 16 - loss : 0.5968 - acc: 0.6709 - val_loss : 0.6019 - val_acc: 0.6620

    100%
    313/313 [02:35<00:00, 2.01it/s]

    Epoch : 17 - loss : 0.5899 - acc: 0.6810 - val_loss : 0.5885 - val_acc: 0.6839

    100%
    313/313 [02:10<00:00, 2.40it/s]

    Epoch : 18 - loss : 0.5908 - acc: 0.6777 - val_loss : 0.5957 - val_acc: 0.6818

    100%
    313/313 [02:09<00:00, 2.42it/s]

    Epoch : 19 - loss : 0.5889 - acc: 0.6808 - val_loss : 0.5925 - val_acc: 0.6822

    100%
    313/313 [02:10<00:00, 2.40it/s]

    Epoch : 20 - loss : 0.5834 - acc: 0.6863 - val_loss : 0.5844 - val_acc: 0.6879


    ```









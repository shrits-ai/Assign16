# Custom U-Net for Image Segmentation

This repository contains an implementation of a custom U-Net model for image segmentation tasks using TensorFlow and Keras. The model supports different configurations and loss functions for training.

## Features
- Custom U-Net architecture with configurable downsampling (MaxPooling or Strided Convolution) and upsampling (Transpose Convolution or UpSampling).
- Supports Binary Cross-Entropy (BCE) and Dice Loss.
- TensorFlow-based data loading and augmentation.
- Automatic GPU/CPU/MPS selection.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- Pillow

Install dependencies using:
```bash
pip install tensorflow tensorflow-metal numpy scikit-learn pillow
```

## File Structure
```
.
├── customUNet.py         # Main script containing model and training logic
├── images/               # Directory containing image-mask pairs
├── README.md             # Project documentation
```

## Training
Run the training script:
```bash
python3 customUNet.py
```
The script will automatically load data from `images/` and train multiple configurations of the U-Net model.

### Training Configurations
The script trains four different model configurations:
1. **MP_Tr_BCE**: MaxPooling + Transpose Convolution with Binary Cross-Entropy loss
2. **MP_Tr_Dice**: MaxPooling + Transpose Convolution with Dice loss
3. **StrConv_Tr_BCE**: Strided Convolution + Transpose Convolution with Binary Cross-Entropy loss
4. **StrConv_Ups_Dice**: Strided Convolution + UpSampling with Dice loss

## Training Logs
The training logs indicate that there are some NaN values in the loss and Dice coefficient. This suggests possible issues with data preprocessing or model stability.
```
python3 customUNet.py 
Using device: /CPU:0
Loading 5912 training pairs...
Loading 1478 validation pairs...

Training MP_Tr_BCE on /CPU:0...
Epoch 1/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4287s 12s/step - accuracy: 0.0026 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 2/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4305s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 3/15
360/370 ━━━━━━━━━━━━━━━━━━━━ 1:48 11s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan^R[ 
  


370/370 ━━━━━━━━━━━━━━━━━━━━ 4307s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 4/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4311s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 5/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4300s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 6/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4284s 12s/step - accuracy: 0.0030 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 7/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4383s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 8/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4313s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 9/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4278s 12s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 10/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4249s 11s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 11/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4246s 11s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 12/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4257s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 13/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4243s 11s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 14/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4242s 11s/step - accuracy: 0.0029 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 15/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4238s 11s/step - accuracy: 0.0029 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Loading 5912 training pairs...
Loading 1478 validation pairs...

Training MP_Tr_Dice on /CPU:0...
Epoch 1/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4322s 12s/step - accuracy: 0.0022 - dice_coef: 1.8392 - loss: -0.8392 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 2/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4340s 12s/step - accuracy: 0.0024 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 3/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4340s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 4/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4337s 12s/step - accuracy: 0.0023 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 5/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4316s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 6/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4321s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 7/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4320s 12s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 8/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4429s 12s/step - accuracy: 0.0021 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 9/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4392s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 10/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4395s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 11/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4870s 13s/step - accuracy: 0.0023 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 12/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4470s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 13/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4574s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 14/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4649s 13s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 15/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4416s 12s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Loading 5912 training pairs...
Loading 1478 validation pairs...

Training StrConv_Tr_BCE on /CPU:0...
Epoch 1/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4568s 12s/step - accuracy: 0.0026 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 2/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4591s 12s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 3/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4570s 12s/step - accuracy: 0.0029 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 4/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4576s 12s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 5/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4574s 12s/step - accuracy: 0.0029 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 6/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4561s 12s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 7/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4574s 12s/step - accuracy: 0.0026 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 8/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4575s 12s/step - accuracy: 0.0027 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 9/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4578s 12s/step - accuracy: 0.0030 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 10/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4569s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 11/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4707s 13s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 12/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4679s 13s/step - accuracy: 0.0030 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 13/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4831s 13s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 14/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 8340s 23s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
Epoch 15/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 4475s 12s/step - accuracy: 0.0028 - dice_coef: nan - loss: nan - val_accuracy: 0.0037 - val_dice_coef: nan - val_loss: nan
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Loading 5912 training pairs...
Loading 1478 validation pairs...

Training StrConv_Ups_Dice on /CPU:0...
Epoch 1/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5158s 14s/step - accuracy: 0.0022 - dice_coef: 1.8707 - loss: -0.8707 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 2/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5264s 14s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 3/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5386s 15s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 4/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5435s 15s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 5/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5298s 14s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 6/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5361s 14s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 7/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5383s 15s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 8/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5370s 15s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 9/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5385s 15s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 10/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5381s 15s/step - accuracy: 0.0023 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 11/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5368s 15s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 12/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5441s 15s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 13/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5572s 15s/step - accuracy: 0.0021 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 14/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5462s 15s/step - accuracy: 0.0022 - dice_coef: 1.9827 - loss: -0.9827 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
Epoch 15/15
370/370 ━━━━━━━━━━━━━━━━━━━━ 5513s 15s/step - accuracy: 0.0022 - dice_coef: 1.9828 - loss: -0.9828 - val_accuracy: 0.0025 - val_dice_coef: 1.9828 - val_loss: -0.9828
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
```




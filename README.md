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
[]
Using device: /GPU:0
2025-02-26 15:27:14.960927: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1 Pro
2025-02-26 15:27:14.960954: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-02-26 15:27:14.960961: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-02-26 15:27:14.960978: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-02-26 15:27:14.960996: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Loading 5912 training pairs...
Unique mask values: [0. 1.]
Loading 1478 validation pairs...

Training MP_Tr_BCE on /GPU:0...
Epoch 1/5
2025-02-26 15:28:50.128347: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
370/370 ━━━━━━━━━━━━━━━━━━━━ 580s 2s/step - accuracy: 0.9961 - dice_coef: 0.9511 - loss: 0.1115 - val_accuracy: 0.9950 - val_dice_coef: 0.9955 - val_loss: 0.0240
Epoch 2/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 558s 2s/step - accuracy: 0.9963 - dice_coef: 0.9953 - loss: 0.0209 - val_accuracy: 0.9950 - val_dice_coef: 0.9955 - val_loss: 0.0199
Epoch 3/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 563s 2s/step - accuracy: 0.9961 - dice_coef: 0.9956 - loss: 0.0193 - val_accuracy: 0.9950 - val_dice_coef: 0.9956 - val_loss: 0.0176
Epoch 4/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 568s 2s/step - accuracy: 0.9961 - dice_coef: 0.9960 - loss: 0.0172 - val_accuracy: 0.9950 - val_dice_coef: 0.9965 - val_loss: 0.0160
Epoch 5/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 563s 2s/step - accuracy: 0.9959 - dice_coef: 0.9963 - loss: 0.0156 - val_accuracy: 0.9950 - val_dice_coef: 0.9954 - val_loss: 0.0150
Loading 5912 training pairs...
Unique mask values: [0. 1.]
Loading 1478 validation pairs...

Training MP_Tr_Dice on /GPU:0...
Epoch 1/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 558s 1s/step - accuracy: 0.9887 - dice_coef: 0.9498 - loss: 0.0502 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 2/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 552s 1s/step - accuracy: 0.9962 - dice_coef: 0.9981 - loss: 0.0019 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 3/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 558s 2s/step - accuracy: 0.9960 - dice_coef: 0.9980 - loss: 0.0020 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 4/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 554s 1s/step - accuracy: 0.9962 - dice_coef: 0.9981 - loss: 0.0019 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 5/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 548s 1s/step - accuracy: 0.9961 - dice_coef: 0.9981 - loss: 0.0019 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Loading 5912 training pairs...
Unique mask values: [0. 1.]
Loading 1478 validation pairs...

Training StrConv_Tr_BCE on /GPU:0...
Epoch 1/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 618s 2s/step - accuracy: 0.9961 - dice_coef: 0.9400 - loss: 0.1356 - val_accuracy: 0.9950 - val_dice_coef: 0.9944 - val_loss: 0.0227
Epoch 2/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 598s 2s/step - accuracy: 0.9962 - dice_coef: 0.9950 - loss: 0.0216 - val_accuracy: 0.9950 - val_dice_coef: 0.9942 - val_loss: 0.0198
Epoch 3/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 602s 2s/step - accuracy: 0.9962 - dice_coef: 0.9954 - loss: 0.0195 - val_accuracy: 0.9950 - val_dice_coef: 0.9954 - val_loss: 0.0172
Epoch 4/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 597s 2s/step - accuracy: 0.9961 - dice_coef: 0.9960 - loss: 0.0168 - val_accuracy: 0.9950 - val_dice_coef: 0.9957 - val_loss: 0.0161
Epoch 5/5
370/370 ━━━━━━━━━━━━━━━━━━━━ 595s 2s/step - accuracy: 0.9964 - dice_coef: 0.9965 - loss: 0.0148 - val_accuracy: 0.9950 - val_dice_coef: 0.9948 - val_loss: 0.0167
Loading 5912 training pairs...
Unique mask values: [0. 1.]
Loading 1478 validation pairs...
Training StrConv_Ups_Dice on /GPU:0...
Epoch 1/5
2025-02-27 19:47:46.693092: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
739/739 ━━━━━━━━━━━━━━━━━━━━ 17238s 23s/step - accuracy: 0.9955 - dice_coef: 0.9668 - loss: 0.0332 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 2/5
739/739 ━━━━━━━━━━━━━━━━━━━━ 17209s 23s/step - accuracy: 0.9962 - dice_coef: 0.9981 - loss: 0.0019 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
Epoch 3/5
739/739 ━━━━━━━━━━━━━━━━━━━━ 17226s 23s/step - accuracy: 0.9962 - dice_coef: 0.9981 - loss: 0.0019 - val_accuracy: 0.9950 - val_dice_coef: 0.9975 - val_loss: 0.0025
```
## Inference from Training Logs

###MP_Tr_BCE

-  Training Dice Coefficient improved from 0.9511 to 0.9963.

-  Validation Dice Coefficient stabilized at 0.9954.

-  Loss reduced from 0.1115 to 0.0150.

-  This configuration demonstrates consistent segmentation performance with binary cross-entropy.

###MP_Tr_Dice

- Training Dice Coefficient reached 0.9981 by Epoch 2 and remained stable.

- Validation Dice Coefficient stabilized at 0.9975.

- Loss dropped significantly to 0.0019.

- This configuration shows that Dice Loss yields better segmentation performance compared to BCE.

###StrConv_Tr_BCE

- Training Dice Coefficient improved from 0.9400 to 0.9965.

- Validation Dice Coefficient reached 0.9957.

- Loss decreased from 0.1356 to 0.0167.

- This model exhibits stable convergence despite slower initial performance.

###StrConv_Ups_Dice

- Training Dice Coefficient improved from 0.9668 to 0.9981.

- Validation Dice Coefficient remained stable at 0.9975.

- Loss consistently dropped to 0.0019.

- Longer training times, this configuration provides the best segmentation performance.

##Conclusion

- StrConv_Ups_Dice and MP_Tr_Dice configurations yield the highest Dice Coefficient and lowest loss, indicating that both configurations achieve similar performance.

- Dice Loss outperforms Binary Cross-Entropy Loss in segmentation quality.

- The model performs well on both training and validation datasets, indicating no major overfitting.





import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image

# Set device based on availability
def get_available_device():
    print(tf.config.list_physical_devices('MPS'))
    if tf.config.list_physical_devices('GPU'):
        return '/GPU:0'  # CUDA-enabled GPU
    elif tf.config.list_physical_devices('MPS'):
        return '/MPS:0'  # macOS MPS
    else:
        return '/CPU:0'  # Fallback to CPU

device = get_available_device()
print(f"Using device: {device}")

# ==============================================================================
#                           Dataset Preparation (Fixed)
# ==============================================================================

def load_custom_dataset(image_dir, mask_dir, img_size=256, test_size=0.2, batch_size=16):
    # Get paired image/mask files with validation
    file_pairs = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):
            mask_path = os.path.join(mask_dir, img_file)
            if os.path.exists(mask_path):
                file_pairs.append((img_file, img_file))
            else:
                print(f"Mask missing for: {img_file}")
    
    if not file_pairs:
        raise ValueError(f"No valid pairs found. Check filenames/extensions.")

    # Split into train/validation
    train_pairs, val_pairs = train_test_split(
        file_pairs, test_size=test_size, random_state=42
    )

    # Load data with progress tracking
    # Load data with progress tracking
    def load_batch(pairs):
        X = np.zeros((len(pairs), img_size, img_size, 3), dtype=np.float32)
        y = np.zeros((len(pairs), img_size, img_size, 1), dtype=np.float32)
        
        for idx, (img_file, mask_file) in enumerate(pairs):
            # Load image
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))
            X[idx] = np.array(img) / 255.0  # Normalize to [0,1]
            
            # Load mask
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path)
            mask = mask.resize((img_size, img_size))
            mask_array = np.array(mask.convert('L'))  # Grayscale (256,256)
            mask_binary = np.where(mask_array > 0, 1, 0).astype(np.float32)
            mask_binary = np.expand_dims(mask_binary, axis=-1)  # Add channel dimension
            y[idx] = mask_binary
        return X, y

    # Load datasets with validation
    print(f"Loading {len(train_pairs)} training pairs...")
    X_train, y_train = load_batch(train_pairs)
    print("Unique mask values:", np.unique(y_train))
    print(f"Loading {len(val_pairs)} validation pairs...")
    X_val, y_val = load_batch(val_pairs)

    # Create TensorFlow datasets
    def augment(image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image, mask

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    
    return train_ds, val_ds

# ==============================================================================
#                           U-Net Components (Same Architecture)
# ==============================================================================

def conv_block(x, filters, kernel_size=(3,3), activation='relu'):
    x = Conv2D(filters, kernel_size, padding='same', activation=activation,
               kernel_initializer='he_normal')(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=activation,
               kernel_initializer='he_normal')(x)
    return x


def build_unet(input_shape=(256,256,3), use_maxpool=True, use_transpose=True):
    inputs = Input(input_shape)
    skip_connections = []
    
    # Encoder
    x = inputs
    for f in [64, 128, 256, 512]:
        x = conv_block(x, f)
        skip_connections.append(x)
        
        # Downsampling
        if use_maxpool:
            x = MaxPooling2D((2, 2))(x)
        else:
            x = Conv2D(f, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    # Bottleneck
    x = conv_block(x, 1024)
    
    # Decoder
    for f in reversed([64, 128, 256, 512]):
        # Upsampling
        if use_transpose:
            x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(x)
        else:
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(f, (2, 2), padding='same', activation='relu')(x)
            
        x = Concatenate()([x, skip_connections.pop()])
        x = conv_block(x, f)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Model(inputs, outputs)

# ==============================================================================
#                           Loss Functions & Metrics
# ==============================================================================


def dice_loss(y_true, y_pred):
    smooth = 1e-6  # Prevents division by zero
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# ==============================================================================
#                           Training Setup
# ==============================================================================

def train_model(config_name, image_dir, mask_dir,
               use_maxpool=True, use_transpose=True,
               loss_fn='binary_crossentropy', epochs=50,
               img_size=256, batch_size=16):
    with tf.device(device):
      # Set float precision
      K.set_floatx('float32')
      
      # Load dataset
      train_ds, val_ds = load_custom_dataset(image_dir, mask_dir, 
                                             img_size=img_size, 
                                             batch_size=batch_size)

      # Build and compile model
      model = build_unet(input_shape=(img_size, img_size, 3),
                        use_maxpool=use_maxpool,
                        use_transpose=use_transpose)
      model.compile(optimizer = Adam(learning_rate=1e-5, clipnorm=1.0),
                     loss=loss_fn,
                     metrics=[dice_coef, 'accuracy'])
      
      print(f"\nTraining {config_name} on {device}...")
      history = model.fit(
         train_ds,
         validation_data=val_ds,
         epochs=epochs
      )
      model.save('my_model.keras')
      return history

# ==============================================================================
#                           Main Execution (Updated)
# ==============================================================================

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    IMAGE_DIR = './images/'  # Should contain .img files
    MASK_DIR = './images'    # Should contain .img masks
    IMG_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 5

    # Verify directory existence
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory {IMAGE_DIR} not found")
    if not os.path.isdir(MASK_DIR):
        raise FileNotFoundError(f"Mask directory {MASK_DIR} not found")

    try:
        # Train all configurations
        configs = [
            ('MP_Tr_BCE', True, True, 'binary_crossentropy'),
            ('MP_Tr_Dice', True, True, dice_loss),
            ('StrConv_Tr_BCE', False, True, 'binary_crossentropy'),
            ('StrConv_Ups_Dice', False, False, dice_loss)
        ]

        for config in configs:
            train_model(config[0], IMAGE_DIR, MASK_DIR,
                      use_maxpool=config[1],
                      use_transpose=config[2],
                      loss_fn=config[3],
                      epochs=EPOCHS,
                      img_size=IMG_SIZE,
                      batch_size=BATCH_SIZE)
    
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Possible solutions:")
        print("1. Verify IMAGE_DIR and MASK_DIR paths")
        print("2. Check files have .img extension")
        print("3. Ensure mask filenames match image filenames")
        print("4. Confirm directories contain valid image/mask pairs")

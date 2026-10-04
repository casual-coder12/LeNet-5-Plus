import tensorflow as tf
import keras
from tensorflow.keras import layers, models


def LeNet5Plus(input_shape, num_classes):
    """
    LeNet-5 Plus model implementation using Keras Functional API.
    This approach provides better compatibility with model saving/loading.
    
    Args:
        input_shape: The shape of the input images (without batch dimension).
        num_classes: The number of classes for the classification task.
    
    Returns:
        A compiled Keras functional model.
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomTranslation(0.05, 0.05),
    ], 
    name='augmentation'
    )

    x = data_augmentation(inputs)

    # C1: Convolutional layer
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        name='C1_Conv'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # S2: Average pooling layer
    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='S2_MaxPool'
    )(x)
    
    # C3: Convolutional layer
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        name='C3_Conv'
    )(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # S4: Average pooling layer
    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='S4_MaxPool'
    )(x)
    
    # C5: Convolutional layer
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        name='C5_Conv'
    )(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Global Average Pooling layer
    x = layers.GlobalAveragePooling2D(name="GAP")(x)
    
    # F6: Fully connected layer
    x = layers.Dense(
        units=84,  
        activation='relu',
        name='F6_Dense'
    )(x)

    # Dropout layer
    x = layers.Dropout(0.5, name='Dropout')(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='Output_Layer'
    )(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LeNet5Plus')
    
    return model
    

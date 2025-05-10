import tensorflow as tf

# Enable mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def build_model(num_classes=29, image_size=(128, 128)):
    # Use EfficientNetB0 with custom settings
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*image_size, 3),
        include_top=False,
        weights='imagenet',
        drop_connect_rate=0.3
    )
    base_model.trainable = False  # Freeze initially

    # Custom head with mixed precision
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax',
        dtype='float32'  # Ensure output is float32
    )(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model
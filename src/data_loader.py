import tensorflow as tf

def load_data(data_dir="../data", image_size=(128, 128), batch_size=64):
    # Training data with augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

   # Validation data (no augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Load training data (1000 images/class)
    train_dataset = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data (1000 images/class)
    val_dataset = val_datagen.flow_from_directory(
        f"{data_dir}/validation",  # Changed from using train/val split
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    # Load test data (1 image/class)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_dataset = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=image_size,
        batch_size=1,  # Important: batch_size=1 for your test set
        class_mode='categorical',
        shuffle=False
    )

    print("\nClass mappings:", train_dataset.class_indices)
    return train_dataset, val_dataset, test_dataset
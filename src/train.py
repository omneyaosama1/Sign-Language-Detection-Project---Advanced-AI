import tensorflow as tf
import numpy as np
from data_loader import load_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Hyperparameters (optimized for 29 classes)
IMAGE_SIZE = (128, 128)  # Balance between speed and accuracy
BATCH_SIZE = 64          # Reduce to 32 if you get memory errors
EPOCHS = 30              # Will stop early if no improvement
INITIAL_LR = 0.001       # Learning rate

def main():
    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Verify class counts
    print("\nTraining samples per class:", np.unique(train_dataset.classes, return_counts=True)[1])
    print("Validation samples per class:", np.unique(val_dataset.classes, return_counts=True)[1])
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        num_classes=len(train_dataset.class_indices),
        image_size=IMAGE_SIZE
    )
    
    # Custom optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath="../models/best_model.h5",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining started...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save("../models/final_model.h5")
    print("\nFinal model saved to models/final_model.h5")
    
    # Evaluate on test set (1 image per class)
    if test_dataset:
        print("\nEvaluating on test set (1 image per class)...")
        test_results = []
        confusion_matrix = np.zeros((29, 29))  # For 29 classes
        
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            pred = model.predict(x, verbose=0)
            true_class = np.argmax(y[0])
            pred_class = np.argmax(pred[0])
            test_results.append(true_class == pred_class)
            confusion_matrix[true_class, pred_class] += 1
            
            # Print individual test results
            class_names = list(train_dataset.class_indices.keys())
            print(f"Sample {i+1}: True={class_names[true_class]}, Predicted={class_names[pred_class]}, "
                  f"Confidence={np.max(pred)*100:.1f}%")
        
        # Print summary
        accuracy = np.mean(test_results)
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        print(f"\nTest Accuracy (29 samples): {accuracy:.2%}")
        
        # Save confusion matrix visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 12))
        plt.imshow(confusion_matrix, cmap='Blues')
        plt.title("Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(29), class_names, rotation=90)
        plt.yticks(range(29), class_names)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("../models/confusion_matrix.png")
        print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    # Configure GPU growth if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()
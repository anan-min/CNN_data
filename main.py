import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau



data_dir = './data'
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=123,
    validation_split=validation_split,
    subset='training',
    interpolation='bilinear',
)


test_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=123,
    validation_split=validation_split,
    subset='validation',
    interpolation='bilinear',
)



def build_model(dropout_rate, l2_rate, learning_rate, epoch):
    try:
        # Load ResNet50 as base model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        # Freeze the first 100 layers of the ResNet50 model
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # Build the CNN model
        CNN_model = keras.Sequential([
            base_model,
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(l2_rate)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(l2_rate)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(20, activation='softmax')
        ])

        CNN_model.summary()

        # Compile the model
        adam = Adam(learning_rate=learning_rate)
        CNN_model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=adam,
                          metrics=["accuracy"])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=6,
            min_lr=1e-5,
            verbose=1
        )

        # Train the model
        history_CNN_model = CNN_model.fit(
            train_data,
            validation_data=test_data,
            epochs=epoch,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )

        # Evaluate the model
        CNN_model.evaluate(test_data)

        return history_CNN_model, CNN_model

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None

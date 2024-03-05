#The model will use the MNIST dataset, contain my own CNN
#http://yann.lecun.com/exdb/mnist/

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
from PIL import Image

class Model:
    def __init__(self):
        self.ds_train, self.ds_test, self._ds_info = self._load_dataset()
        print("Dataset loaded!")
        self._preprocess_data()
        print("Preprocessing done!")
        self.models = {}
        self._initialize_models()


    def _load_dataset(self):
        (self.ds_train, self.ds_test), self.ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        return self.ds_train, self.ds_test, self.ds_info
    
    def _normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def _preprocess_data(self):
        self.ds_train = self.ds_train.map(self._normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle(self.ds_info.splits['train'].num_examples)
        self.ds_train = self.ds_train.batch(128)
        self.ds_train = self.ds_train.prefetch(tf.data.AUTOTUNE)

        self.ds_test = self.ds_test.map(
        self._normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_test = self.ds_test.batch(128)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.prefetch(tf.data.AUTOTUNE)

    def _initialize_models(self):
        tf_docs_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        #print("tf_docs")
        small_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        #print("small")
        medium_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        #print("medium")
        large_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        #print("large")
        self.models = {
        "tf_docs": tf_docs_model,
        "small": small_model,
        "medium": medium_model,
        "large": large_model
        }

    def custom_cnn(self, model_name="medium"):
        
        
        self.models[model_name].compile(
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        self.models[model_name].fit(
            self.ds_train,
            epochs=6,
            validation_data=self.ds_test,
        )

        print("Model Trained!")

        self.models[model_name].summary()

        path = os.path.join(os.getcwd(), "out")

        if not os.path.exists(path):
            os.makedirs(path)
        
        path = os.path.join(path,  model_name+".keras")

        self.models[model_name].save(path)
    
    def preprocess_image(self, image_path):
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST image size
        img_array = np.array(img)  # Convert to numpy array
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def predict(self, model_name, images):
        # Load model
        model_path = os.path.join(os.getcwd(), "out", f"{model_name}.keras")
        model = tf.keras.models.load_model(model_path)

        # Preprocess images
        images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.

        # Make predictions
        predictions = model.predict(images)

        return predictions
    
    def print_prediction(self, prediction):
        predicted_class = np.argmax(prediction)
        print("Predicted class:", predicted_class)
        print("Confidence:", prediction[0][predicted_class])

    
    

def main():
    model_name = "tf_docs"

    model = Model()
    model.custom_cnn(model_name)

    image_path = "" #add the path for the image that you want the modil to predict

    if not os.path.exists(image_path):
        print("Image file not found!")
        return
    
    # Preprocess image
    image_array = model.preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(model_name, image_array)
    
    # Print prediction
    model.print_prediction(prediction)



if __name__ == "__main__":
    main()
#The model will use the MNIST dataset, contain my own CNN
#http://yann.lecun.com/exdb/mnist/


import tensorflow as tf
import tensorflow_datasets as tfds
import os

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
            optimizer=tf.keras.optimizers.Adam(0.001),
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

    
    

def main():
    model = Model()
    model.custom_cnn("medium")


if __name__ == "__main__":
    main()
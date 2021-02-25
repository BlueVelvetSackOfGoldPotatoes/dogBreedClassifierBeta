import os
import re
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

model_name = 'ResNet50'
activation_function = 'relu',

dataset, info = tfds.load(name="stanford_dogs", with_info=True)

training_data = dataset['train']
test_data = dataset['test']

# Constants
IMG_LEN = 224
IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
N_BREEDS = 120

def save_convert_model(model):
    # Save the full model
    model.save('savedModels/' + model_name)

    concrete_func = model.__call__.get_concrete_function()

    # Convert to lite use on mobile
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = conerter.convert()

    # Save the lite version of the model
    with open('savedModels' + model_name + '_model.tflite', 'wb') as f:
        f.write(tflite_model)

def preprocess(ds_row):
    # Image conversion int->float + resizing
    image = tf.image.convert_image_dtype(ds_row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')
  
    # Onehot encoding labels
    label = tf.one_hot(ds_row['label'],N_BREEDS)

    return image, label

def prepare(dataset, batch_size=None):
    ds = dataset.map(preprocess, num_parallel_calls=4)
    ds = ds.shuffle(buffer_size=1000)
    if batch_size:
      ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def classifier():
    # Using mobile friendly network without fully connected layer for transfer learning

    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               pooling=max)
    # Freeze model
    base_model.trainable = False

    # Add two layers: GlobalAveragePooling2D to transform the above tensor into a vector and a single dense layer for predicting output classes
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(N_BREEDS, activation='relu')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adamax(0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    # Output 1
    model.summary()  

    train_batches = prepare(training_data, batch_size=32)
    test_batches = prepare(test_data, batch_size=32)

    history = model.fit(train_batches,
                        epochs=20,
                        validation_data=test_batches)

    save_convert_model(model)

    # Output 2
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Test accuracy')
    plt.legend()
    plt.savefig('accuracyGraphs/' + model_name + '_accuracy.svg')
    plt.show()

def main():
    print("Running...\n")
    classifier()
    
if __name__== "__main__" :
    main()
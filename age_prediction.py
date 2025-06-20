import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, brightness_range=[0.7, 1.3])
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('data/age_train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_gen = test_datagen.flow_from_directory('data/age_test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(95, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=5, validation_data=test_gen)

# Fine-tuning phase
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=5, validation_data=test_gen)

model.save('models/age_prediction_model.h5')

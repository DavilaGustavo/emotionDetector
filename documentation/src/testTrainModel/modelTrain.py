import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports for training
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Load the data
data = pd.read_csv('data/fer2013/fer2013.csv')
print(data.tail())

# Plot a graph showing the total number of images corresponding to each emotion
plt.figure(figsize=(12,6))
plt.hist(data['emotion'], bins=6)
plt.title("Images x Emotion")
plt.show()
# Classes: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Process the data
pixels = data['pixels'].tolist()
width, height = 48, 48

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height) 
    faces.append(face)

faces = np.asarray(faces) 
faces = np.expand_dims(faces, -1)

def normalize(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

faces = normalize(faces)
emotions = pd.get_dummies(data['emotion']).to_numpy()

print("Total number of images in the dataset: " + str(len(faces)))

# Split into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=41)

print("Number of images in the training set:", len(x_train))
print("Number of images in the test set:", len(x_test))
print("Number of images in the validation set:", len(y_val))

# Save data for testing
np.save('mod_xtest', x_test)
np.save('mod_ytest', y_test)

# Model construction
num_features = 32
num_classes = 7
width, height = 48, 48
batch_size = 16
epochs = 100

model = Sequential()

model.add(Conv2D(num_features, (3, 3), padding='same', kernel_initializer="he_normal",
                 input_shape=(width, height, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

model_weights_file = "model_expressions.keras"  # Model file
model_json_file = "model_expressions.json"  # JSON file to save the architecture
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(model_weights_file, monitor='val_loss', verbose=1, save_best_only=True)

# Save the model
model_json = model.to_json()
with open(model_json_file, "w") as json_file:
    json_file.write(model_json)

# Train the model
history = model.fit(np.array(x_train), np.array(y_train),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(np.array(x_val), np.array(y_val)),
        shuffle=True,
        callbacks=[lr_reducer, early_stopper, checkpointer])

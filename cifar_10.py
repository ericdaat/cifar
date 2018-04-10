# coding: utf-8
import pickle
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def parse_image(row):
    r = row[:1024]
    g = row[1024:2048]
    b = row[2048:]

    return np.asarray([r, g, b]).T.reshape(32, 32, 3)


data = []
labels = []

for fname in glob('./cifar-10-batches-py/data_batch_[0-9]'):
    pkl = unpickle(fname)
    data_ = pkl[b'data']
    labels_ = np.asarray(pkl[b'labels']).reshape(-1, 1)

    data.append(np.asarray(list(map(parse_image, data_))))
    labels.append(np.asarray(labels_))
else:
    X = np.vstack(np.asarray(data))
    y = np.vstack(np.asarray(labels))

del data
del labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

pkl = unpickle('./cifar-10-batches-py/test_batch')
X_test = np.asarray(list(map(parse_image, pkl[b'data'])))
y_test = np.asarray(pkl[b'labels']).reshape(-1, 1)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same',))
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=0.0001)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['sparse_categorical_accuracy'])

# tensorboard = TensorBoard(log_dir='/home/manomano/tensorboard-logs/cifar-10')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              min_lr=0.00001)

model.fit(X_train,
          y_train,
          validation_data=(X_val, y_val),
          batch_size=32,
          epochs=80,
          callbacks=[reduce_lr])

score = model.evaluate(X_test, y_test, batch_size=32)
print('Test Loss is {0:.3f}'.format(score[0]))
print('Test Accuracy is {0:.3f}'.format(score[1]))

model.save('/home/eric/code/image_search/application/models/cifar-10.h5')

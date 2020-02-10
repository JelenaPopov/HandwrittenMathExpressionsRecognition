import numpy as np
import glob
import cv2
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator, img_to_array

from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold, train_test_split
from keras.models import model_from_json

symbols = [
    '-', '!', '(', ')', ',', '[', ']', '{', '}', '+', '=', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', 'a', 'alpha', 'b', 'beta', 'c', 'cos',
    'd', 'div', 'e', 'exists', 'f', 'forall', '/', 'g', 'gamma', 'geq', '>', 'h', 'i',
    'in', 'infty', 'int', 'j', 'k', 'ldots', 'leq', 'lim', 'log', '<', 'm', 'n',
    'neq', 'p', 'phi', 'pi', 'pm', 'r', 'rightarrow', 'sin', 'sqrt', 'sum',
    't', 'tan', 'theta', 'times', 'x', 'y', 'z'
]

folderName = [
    '-', '!', '(', ')', ',', '[', ']', '{', '}', '+', '=', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', 'a', 'alpha', 'b', 'beta', 'c', 'cos',
    'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'g', 'gamma', 'geq', 'gt', 'h', 'i',
    'in', 'infty', 'int', 'j', 'k', 'ldots', 'leq', 'lim', 'log', 'lt', 'm', 'n',
    'neq', 'p', 'phi', 'pi', 'pm', 'r', 'rightarrow', 'sin', 'sqrt', 'sum',
    't', 'tan', 'theta', 'times', 'x', 'y', 'z'
]


def scale_to_range(X):
    X = X.astype('float32')
    return X / 255.0


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def convert_output(symbols):
    nn_outputs = []
    for index in range(len(symbols)):
        output = np.zeros(len(symbols))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def serialize_ann(ann):
    model_json = ann.to_json()
    with open("serialization_folder/trained_model.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights("serialization_folder/trained_model.h5")


def load_trained_ann():
    try:
        json_file = open('serialization_folder/trained_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights("serialization_folder/trained_model.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        return None


def erode(image, num_of_iteration):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=num_of_iteration)


def load_training_data(training_directory):
    image_data = []
    label_data = []

    i = 0
    n = 0
    symbols_array_of_zero_and_one = convert_output(symbols)
    print(len(symbols))
    print(len(folderName))
    for iterator in range(0, len(symbols)):
        print(training_directory + "/" + folderName[iterator])
        print(i)
        for img in glob.glob(training_directory + "/" + folderName[iterator] + "/*.jpg"):
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_RGB2GRAY)
            erode_img = erode(image, 1)
            resized_image = cv2.resize(image, (28, 28))
            image_data.append(resized_image)
            image_data.append(cv2.resize(erode_img, (28, 28)))
            label_data.append(symbols_array_of_zero_and_one[i])
            label_data.append(symbols_array_of_zero_and_one[i])
            n = n + 2
        i = i + 1
        print(n)
        n = 0
    x_train, x_test, y_train, y_test = train_test_split(image_data, label_data, test_size=0.20, random_state=27)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(69, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, trainY, x_test, y_test):
    model = define_model()

    model.fit(trainX, trainY, validation_data=(x_test, y_test), epochs=10, batch_size=32, verbose=1)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))
    return model


def load_or_save_train_data(training_directory):
    try:
        x_train = np.load('serialization_folder/train_data.npy')
        y_train = np.load('serialization_folder/label_data.npy')
        x_test = np.load('serialization_folder/x_test.npy')
        y_test = np.load('serialization_folder/y_test.npy')
        return x_train, y_train, x_test, y_test
    except IOError:
        (X_train, Y_train, x_test, y_test) = load_training_data(training_directory)
        np.save('serialization_folder/train_data', X_train)
        np.save('serialization_folder/label_data', Y_train)
        np.save('serialization_folder/x_test', x_test)
        np.save('serialization_folder/y_test', y_test)
    return X_train, Y_train, x_test, y_test


def prepare_load_data(training_directory):
    (x_train, Y_train, x_test, y_test) = load_or_save_train_data(training_directory)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
    # normalize inputs from 0-255 to 0-1
    x_train = scale_to_range(x_train)
    x_test = scale_to_range(x_test)
    return x_train, Y_train, x_test, y_test


def prepare_data_for_prediction(x_test):
    x_test = np.array(x_test)
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
    x_test = scale_to_range(x_test)
    return x_test


def prepare_data_for_prediction_lstm(x_test):
    x_test = np.array(x_test)
    x_test = x_test.reshape((x_test.shape[0], 28, 28)).astype('float32')
    x_test = scale_to_range(x_test)
    return x_test


def run_train_model():
    trained_model = load_trained_ann()
    if trained_model is None:
        x_train, y_train, x_test, y_test = prepare_load_data("dataset/train/extracted_images")
        print("Treniranje modela zapoceto.")
        trained_model = evaluate_model(x_train, y_train, x_test, y_test)
        print("Treniranje modela zavrseno.")
        serialize_ann(trained_model)
        print('\n# Evaluate on test data')
        results = trained_model.evaluate(np.array(x_test, np.float), y_test, batch_size=128)
        print('test loss, test acc:', results)
    return trained_model


def predict(trained_model, x_test):
    label_array = []
    results = trained_model.predict(np.array(x_test, np.float))
    for result in results:
        win = symbols[winner(result)]
        label_array.append(win)
    return label_array

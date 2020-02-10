import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import inkmlToPng
from LSTM import run_train_model_lstm, predict_lstm
from MathExpressions import get_math_expression_from_prediction_result
from TrainRecognizeMathSymbols import prepare_data_for_prediction, run_train_model, predict, \
    prepare_data_for_prediction_lstm

folderName = [
    '-', '!', '(', ')', ',', '[', ']', '{', '}', '+', '=', '0', '1', '2',
    '3', '4', '5', '6', '7', '8', '9', 'a', 'alpha', 'b', 'beta', 'c', 'cos',
    'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'g', 'gamma', 'geq', 'gt', 'h', 'i',
    'in', 'infty', 'int', 'j', 'k', 'ldots', 'leq', 'lim', 'log', 'lt', 'm', 'n',
    'neq', 'p', 'phi', 'pi', 'pm', 'r', 'rightarrow', 'sin', 'sqrt', 'sum',
    't', 'tan', 'theta', 'times', 'x', 'y', 'z'
]


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def get_bounding_boxes(formula):
    # display_image(formula)
    ret, thresh = cv2.threshold(formula, 0, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    id_c = 0
    im = formula.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append({
            'id': id_c,
            'xmin': x,
            'xmax': x + w,
            'ymin': y,
            'ymax': y + h,
            'combined': [],
            'paint': False
        })
        id_c += 1
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("ssssssssss1111ss", im)
    bounding_boxes = sorted(bounding_boxes, key=lambda k: (k['xmin'], k['ymin']))
    new_bounding_boxes = join_contours(bounding_boxes)
    # im = formula.copy()
    # cv2.imshow("ssssss1111ss", im)

    for box in new_bounding_boxes:
        cv2.rectangle(im, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 0, 255), 2)
    # cv2.imshow("sss", im)
    # cv2.waitKey(0)
    ret, thresh = cv2.threshold(formula, 0, 255, cv2.THRESH_BINARY)
    regions_array, img = made_regions(thresh, new_bounding_boxes)
    return regions_array


def join_contours(bounding_boxes):
    new_bounding_boxes = []
    id_c = 0
    index = 0
    j = 1
    z = 2
    size_of_bounding_boxes = len(bounding_boxes)
    while index < size_of_bounding_boxes:
        j = index + 1
        z = index + 2
        if j < size_of_bounding_boxes and z < size_of_bounding_boxes:
            first_contour_width = bounding_boxes[index]['xmax'] - bounding_boxes[index]['xmin']
            second_contour_width = bounding_boxes[j]['xmax'] - bounding_boxes[j]['xmin']
            third_contour_width = bounding_boxes[z]['xmax'] - bounding_boxes[z]['xmin']

            first_contour_height = bounding_boxes[index]['ymax'] - bounding_boxes[index]['ymin']
            second_contour_height = bounding_boxes[j]['ymax'] - bounding_boxes[j]['ymin']
            third_contour_height = bounding_boxes[z]['ymax'] - bounding_boxes[z]['ymin']
            max_y_values = []
            max_y_values.append(bounding_boxes[index]['ymax'])
            max_y_values.append(bounding_boxes[j]['ymax'])
            max_y_values.append(bounding_boxes[z]['ymax'])
            max_val = max(max_y_values)
            ymin_values = []
            ymin_values.append(bounding_boxes[index]['ymin'])
            ymin_values.append(bounding_boxes[j]['ymin'])
            ymin_values.append(bounding_boxes[z]['ymin'])
            if int(first_contour_width / 3) >= second_contour_width and int(
                    first_contour_width / 3) >= third_contour_width and first_contour_height < 15:
                new_bounding_boxes.append({
                    'id': id_c,
                    'xmin': bounding_boxes[index]['xmin'],
                    'xmax': bounding_boxes[index]['xmax'],
                    'ymin': min(ymin_values),
                    'ymax': max_val,
                    'combined': [],
                    'paint': False
                })
                index += 3
                id_c += 1
                continue
            # for  ...
            elif first_contour_height < 11 and second_contour_height < 11 and third_contour_height < 11 and \
                    first_contour_width < 11 and second_contour_width < 11 and third_contour_width < 11:
                new_bounding_boxes.append({
                    'id': id_c,
                    'xmin': bounding_boxes[index]['xmin'],
                    'xmax': bounding_boxes[z]['xmax'],
                    'ymin': min(ymin_values),
                    'ymax': max_val,
                    'combined': [],
                    'paint': False
                })
                index += 3
                id_c += 1
                continue
        # for +-, <= and =
        if j < size_of_bounding_boxes:

            max_y_values = []
            max_y_values.append(bounding_boxes[index]['ymax'])
            max_y_values.append(bounding_boxes[j]['ymax'])
            max_val = max(max_y_values)
            max_x_values = []
            max_x_values.append(bounding_boxes[index]['xmax'])
            max_x_values.append(bounding_boxes[j]['xmax'])
            min_y_values = []
            min_y_values.append(bounding_boxes[index]['ymin'])
            min_y_values.append(bounding_boxes[j]['ymin'])
            new_x_max = bounding_boxes[index]['xmin'] + 2 / 3 * (
                    bounding_boxes[index]['xmax'] - bounding_boxes[index]['xmin'])
            # sqrt
            if bounding_boxes[index]['xmin'] <= bounding_boxes[j]['xmin'] <= bounding_boxes[index]['xmax'] and \
                    bounding_boxes[index]['xmin'] <= bounding_boxes[j]['xmax'] <= bounding_boxes[index]['xmax'] and \
                    bounding_boxes[index]['ymin'] <= bounding_boxes[j]['ymin'] <= bounding_boxes[index]['ymax'] and \
                    bounding_boxes[index]['ymin'] <= bounding_boxes[j]['ymax'] <= (bounding_boxes[index]['ymax'] + 2):
                new_bounding_boxes.append({
                    'id': id_c,
                    'xmin': bounding_boxes[index]['xmin'],
                    'xmax': bounding_boxes[index]['xmax'],
                    'ymin': bounding_boxes[index]['ymin'],
                    'ymax': bounding_boxes[index]['ymax'],
                    'combined': [],
                    'paint': True
                })
                id_c += 1
                index += 1
                continue
            # for +-, <= and =
            elif bounding_boxes[index]['xmin'] <= bounding_boxes[j]['xmin'] <= new_x_max:
                if z < size_of_bounding_boxes:
                    # 9/2
                    if bounding_boxes[index]['xmin'] <= bounding_boxes[z]['xmin'] <= bounding_boxes[index]['xmax']:
                        new_bounding_boxes.append({
                            'id': id_c,
                            'xmin': bounding_boxes[index]['xmin'],
                            'xmax': bounding_boxes[index]['xmax'],
                            'ymin': bounding_boxes[index]['ymin'],
                            'ymax': bounding_boxes[index]['ymax'],
                            'combined': [],
                            'paint': False
                        })
                        id_c += 1
                        new_bounding_boxes.append({
                            'id': id_c,
                            'xmin': bounding_boxes[j]['xmin'],
                            'xmax': bounding_boxes[j]['xmax'],
                            'ymin': bounding_boxes[j]['ymin'],
                            'ymax': bounding_boxes[j]['ymax'],
                            'combined': [],
                            'paint': False
                        })
                        id_c += 1
                        new_bounding_boxes.append({
                            'id': id_c,
                            'xmin': bounding_boxes[z]['xmin'],
                            'xmax': bounding_boxes[z]['xmax'],
                            'ymin': bounding_boxes[z]['ymin'],
                            'ymax': bounding_boxes[z]['ymax'],
                            'combined': [],
                            'paint': False
                        })
                        id_c += 1
                        index += 3
                        continue
                    else:
                        new_bounding_boxes.append({
                            'id': id_c,
                            'xmin': bounding_boxes[index]['xmin'],
                            'xmax': max(max_x_values),
                            'ymin': min(min_y_values),
                            'ymax': max_val,
                            'combined': [],
                            'paint': False
                        })
                        index += 2
                        id_c += 1
                        continue
                else:
                    # for +-, <= and =
                    new_bounding_boxes.append({
                        'id': id_c,
                        'xmin': bounding_boxes[index]['xmin'],
                        'xmax': max(max_x_values),
                        'ymin': min(min_y_values),
                        'ymax': max_val,
                        'combined': [],
                        'paint': False
                    })
                    index += 2
                    id_c += 1
                    continue
        first_contour_height = bounding_boxes[index]['ymax'] - bounding_boxes[index]['ymin']
        first_contour_width = bounding_boxes[index]['xmax'] - bounding_boxes[index]['xmin']
        if first_contour_height >= 6 or first_contour_width > 5:
            new_bounding_boxes.append({
                'id': id_c,
                'xmin': bounding_boxes[index]['xmin'],
                'xmax': bounding_boxes[index]['xmax'],
                'ymin': bounding_boxes[index]['ymin'],
                'ymax': bounding_boxes[index]['ymax'],
                'combined': [],
                'paint': False
            })
            id_c += 1
            index += 1
            continue
        index += 1
    return new_bounding_boxes


def made_regions(img, bounding_boxes):
    regions_array = []
    i = 1
    for box in bounding_boxes:
        if i >= len(bounding_boxes):
            i = i - 1
        region = cv2.resize(made_region(box, bounding_boxes[i], img), (28, 28))
        # display_image(region)
        regions_array.append([region, box])
        i += 1
    return regions_array, img


def erode(image, num_of_iteration):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=num_of_iteration)


def made_region(box, next_box, img):
    region_width = box['xmax'] - box['xmin']
    region_height = box['ymax'] - box['ymin']

    if region_height == region_width:
        return img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]

    if region_width < region_height:
        if region_height < 28:
            new_image = np.zeros((28, 28))
            new_image[:] = 255
            difference_width = 28 - region_width
            difference_height = 28 - region_height
            new_image[int(difference_height / 2):(int(difference_height / 2 + region_height)),
            int(difference_width / 2):(int(difference_width / 2 + region_width))] = \
                img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
            # plt.imshow(new_image, cmap="gray")
            # plt.show()
            return new_image
        else:
            new_image = np.ones((region_height, region_height))
            new_image[:] = 255
            difference = region_height - region_width
            new_image[0:region_height, int(difference / 2):(int(difference / 2 + region_width))] = \
                img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
            # plt.subplot(1, 2, 1)
            # plt.imshow(new_image, cmap="gray")
            if region_height >= 150:
                new_image = erode(new_image, 2)
            if region_height >= 100:
                new_image = erode(new_image, 1)
            # plt.subplot(1, 2, 2)
            # plt.imshow(new_image, cmap="gray")
            # plt.show()
            return new_image
    if region_width > region_height:
        if region_width < 28:
            new_image = np.zeros((28, 28))
            new_image[:] = 255
            difference_width = 28 - region_width
            difference_height = 28 - region_height
            new_image[int(difference_height / 2):(int(difference_height / 2 + region_height)),
            int(difference_width / 2):(int(difference_width / 2 + region_width))] = \
                img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
            # plt.imshow(new_image, cmap="gray")
            # plt.show()
            return new_image
        else:
            new_image = np.zeros((region_width, region_width))
            new_image[:] = 255
            difference = region_width - region_height
            new_image[int(difference / 2): (int(difference / 2) + region_height), 0:region_width] = \
                img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
            if box['paint']:
                new_image[int(difference / 2) + 10: (int(difference / 2) + region_height),
                next_box['xmin'] - box['xmin']:] = 255
            # plt.subplot(1, 2, 1)
            # plt.imshow(new_image, cmap="gray")
            if region_width >= 200:
                new_image = erode(new_image, 2)
            if region_width >= 100:
                new_image = erode(new_image, 1)
            # plt.subplot(1, 2, 2)
            # plt.imshow(new_image, cmap="gray")
            # plt.show()

            return new_image


def load_png_images(img_path, trained_model, trained_model_lstm):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # display_image(img)
    regions_and_box_array = get_bounding_boxes(img)

    # CNN
    regions_array_for_prediction = [region[0] for region in regions_and_box_array]
    regions_array_for_prediction = prepare_data_for_prediction(regions_array_for_prediction)
    bounding_boxes = [region[1] for region in regions_and_box_array]
    results = predict(trained_model, regions_array_for_prediction)
    expression = get_math_expression_from_prediction_result(results, bounding_boxes)

    # LSTM
    regions_array_for_prediction = [region[0] for region in regions_and_box_array]
    regions_array_for_prediction_lstm = prepare_data_for_prediction_lstm(regions_array_for_prediction)
    bounding_boxes = [region[1] for region in regions_and_box_array]
    results_lstm = predict_lstm(trained_model_lstm, regions_array_for_prediction_lstm)
    expression_lstm = get_math_expression_from_prediction_result(results_lstm, bounding_boxes)

    return results, expression, results_lstm, expression_lstm


def load_tests_one_simple_expression(test_directory):
    truth_labels_for_expressions = []
    truth_labels_for_symbols = []
    predicted_labels_for_expressions = []
    predicted_labels_for_symbols = []
    predicted_labels_for_expressions_lstm = []
    predicted_labels_for_symbols_lstm = []
    truth_symbols_dictionary, truth_expression_dictionary = inkmlToPng.read_label_from_txt_file(
        "dataset/test/png/label.txt")
    trained_model = run_train_model()
    trained_model_lstm = run_train_model_lstm()
    ratio_cnn = 0
    ratio_lstm = 0
    for img_name in os.listdir("dataset/test/png"):
        if img_name.find("png") != -1:
            img_path = os.path.join(test_directory, img_name)
            results, expression, results_lstm, expression_lstm = load_png_images(img_path, trained_model,
                                                                                 trained_model_lstm)

            predicted_labels_for_expressions.append(expression)
            predicted_labels_for_symbols.append(results)

            predicted_labels_for_expressions_lstm.append(expression_lstm)
            predicted_labels_for_symbols_lstm.append(results_lstm)

            truth_labels_for_expressions.append(truth_expression_dictionary[img_path])
            truth_labels_for_symbols.append(truth_symbols_dictionary[img_path])
            ratio_cnn += fuzz.ratio(expression, truth_expression_dictionary[img_path])
            ratio_lstm += fuzz.ratio(expression_lstm, truth_expression_dictionary[img_path])
            print("TEST IMAGES WITH ONE EXPRESSION CNN")
            print(img_path)
            print("Thruth expression")
            print(truth_expression_dictionary[img_path])
            print("Predicted expression")
            print(expression)
            print("FuzzyWuzzy ratio")
            print(fuzz.ratio(expression, truth_expression_dictionary[img_path]))
            print("*****************************")

            print("TEST IMAGES WITH ONE EXPRESSION LSTM ")
            print(img_path)
            print("Thruth expression")
            print(truth_expression_dictionary[img_path])
            print("Predicted expression")
            print(expression_lstm)
            print("FuzzyWuzzy ratio")
            print(fuzz.ratio(expression_lstm, truth_expression_dictionary[img_path]))
            print("*****************************")
    print("TEST IMAGES WITH ONE EXPRESSION CNN")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols, truth_labels_for_expressions,
                 predicted_labels_for_expressions)
    print("FuzzyWazzy average ratio")
    print(round(ratio_cnn / len(truth_labels_for_expressions), 2))
    print("********************************")
    print("TEST IMAGES WITH ONE EXPRESSION LSTM ")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols_lstm, truth_labels_for_expressions,
                 predicted_labels_for_expressions_lstm)
    print("FuzzyWazzy average ratio")
    print(round(ratio_lstm / len(truth_labels_for_expressions), 2))
    print("********************************")


def get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols, truth_labels_for_expressions,
                 predicted_labels_for_expressions):
    total_number_of_symbols = 0
    total_number_of_truth_predicted_symbols = 0
    for i in range(0, len(truth_labels_for_symbols)):
        truth_labels = truth_labels_for_symbols[i]
        predicted_labels = predicted_labels_for_symbols[i]
        if len(truth_labels) <= len(predicted_labels):
            count = len(["ok" for idx, label in enumerate(truth_labels) if label == predicted_labels[idx]])
            total_number_of_truth_predicted_symbols += count
            total_number_of_symbols += len(truth_labels)
        else:
            count = len(["ok" for idx, label in enumerate(predicted_labels) if label == truth_labels[idx]])
            total_number_of_truth_predicted_symbols += count
            total_number_of_symbols += len(truth_labels)
    print("Results -> prediction math expressions")
    score_expressions = accuracy_score(truth_labels_for_expressions, predicted_labels_for_expressions)
    print('Accuracy:', round(score_expressions * 100))


# inkmlToPng.load_inkml_and_save_to_png("dataset/test/one_expression")
load_tests_one_simple_expression("dataset/test/png/")

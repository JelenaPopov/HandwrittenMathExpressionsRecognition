import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import fuzz
import inkmlToPng
from LSTM import predict_lstm, run_train_model_lstm
from MathExpressions import get_math_expression_from_prediction_result
from TestOneSimpleExpression import made_regions, get_accuracy
from TrainRecognizeMathSymbols import prepare_data_for_prediction, run_train_model, predict, \
    prepare_data_for_prediction_lstm


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


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
            if int(first_contour_width / 2) >= second_contour_width and int(
                    first_contour_width / 2) >= third_contour_width and first_contour_height < 15:
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
                added = True
                continue
            # for +-, <= and =
            elif bounding_boxes[index]['xmin'] <= bounding_boxes[j]['xmin'] <= new_x_max and (
                    bounding_boxes[index]['xmax'] - bounding_boxes[j]['xmin']) >= (
                    bounding_boxes[j]['xmax'] - bounding_boxes[j]['xmin']) * 0.5:
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
        if first_contour_height >= 17 or first_contour_width > 17:
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


def load_rotated_image_and_return_normal_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # display_image(image)
    gray = cv2.bitwise_not(image)
    img = gray.copy()
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    coords = np.column_stack(np.where(thresh > 0))
    rotated_rect = cv2.minAreaRect(coords)
    (x, y), (width, height), angle = rotated_rect

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if 1 > (image.shape[0] - image.shape[1]) > -17 and (width - height) > 0:
        angle = -angle
    elif (image.shape[0] - image.shape[1]) > 0 and (width - height) > 0:
        return image

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255), borderMode=cv2.BORDER_CONSTANT)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(rotated, cmap="gray")
    # plt.show()
    return rotated


def dilate(image, num_of_iteration):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=num_of_iteration)


def get_bounding_boxes_for_rotated_image(formula):
    # thresh = cv2.bitwise_not(formula)
    _, thresh = cv2.threshold(formula, 125, 255, cv2.THRESH_BINARY_INV)
    # display_image(thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    id_c = 0
    # im = formula.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2:
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
    #         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # display_image(im)
    bounding_boxes = sorted(bounding_boxes, key=lambda k: (k['xmin'], k['ymin']))
    new_bounding_boxes = join_contours(bounding_boxes)
    regions_array, img = made_regions(formula, new_bounding_boxes)
    return regions_array


def get_prediction_results(img, trained_model, trained_model_lstm):
    regions_and_box_array = get_bounding_boxes_for_rotated_image(img)

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


def test_rotated_images():
    truth_labels_for_expressions = []
    truth_labels_for_symbols = []
    predicted_labels_for_expressions = []
    predicted_labels_for_symbols = []
    predicted_labels_for_expressions_lstm = []
    predicted_labels_for_symbols_lstm = []
    truth_symbols_dictionary, truth_expression_dictionary = inkmlToPng.read_label_from_txt_file(
        "dataset/test/rotated_image/label.txt")
    trained_model = run_train_model()
    trained_model_lstm = run_train_model_lstm()
    ratio_cnn = 0
    ratio_lstm = 0
    for img_name in os.listdir("dataset/test/rotated_image"):
        if img_name.find("png") != -1:
            img_path = "dataset/test/rotated_image/" + img_name

            img = load_rotated_image_and_return_normal_image(img_path)
            results, expression, results_lstm, expression_lstm = get_prediction_results(img, trained_model,
                                                                                        trained_model_lstm)
            predicted_labels_for_expressions.append(expression)
            predicted_labels_for_symbols.append(results)

            predicted_labels_for_expressions_lstm.append(expression_lstm)
            predicted_labels_for_symbols_lstm.append(results_lstm)

            truth_labels_for_expressions.append(truth_expression_dictionary[img_path])
            truth_labels_for_symbols.append(truth_symbols_dictionary[img_path])
            ratio_cnn += fuzz.ratio(expression, truth_expression_dictionary[img_path])
            ratio_lstm += fuzz.ratio(expression_lstm, truth_expression_dictionary[img_path])

            print("TEST IMAGES WITH ROTATED EXPRESSION CNN")
            print(img_path)
            print("Thruth expression")
            print(truth_expression_dictionary[img_path])
            print("Predicted expression")
            print(expression)
            print("FuzzyWuzzy ratio")
            print(fuzz.ratio(expression, truth_expression_dictionary[img_path]))
            # print("Thruth symbols")
            # print(truth_symbols_dictionary[img_path])
            # print("Predicted symbols: ")
            # print(results)
            print("*****************************")

            print("TEST IMAGES WITH ONE ROTATED LSTM ")
            print(img_path)
            print("Thruth expression")
            print(truth_expression_dictionary[img_path])
            print("Predicted expression")
            print(expression_lstm)
            print("FuzzyWuzzy ratio")
            print(fuzz.ratio(expression_lstm, truth_expression_dictionary[img_path]))
            # print("Thruth symbols")
            # print(truth_symbols_dictionary[img_path])
            # print("Predicted symbols: ")
            # print(results_lstm)
            print("*****************************")
    print("TEST IMAGES WITH ONE ROTATED EXPRESSION CNN")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols, truth_labels_for_expressions,
                 predicted_labels_for_expressions)
    print("FuzzyWazzy average ratio")
    print(round(ratio_cnn / len(truth_labels_for_expressions), 2))
    print("********************************")
    print("TEST IMAGES WITH ONE ROTATED EXPRESSION LSTM ")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols_lstm, truth_labels_for_expressions,
                 predicted_labels_for_expressions_lstm)
    print("FuzzyWazzy average ratio")
    print(round(ratio_lstm / len(truth_labels_for_expressions), 2))
    print("********************************")

# test_rotated_images()

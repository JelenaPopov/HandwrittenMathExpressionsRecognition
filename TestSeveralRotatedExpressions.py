import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from LSTM import predict_lstm, run_train_model_lstm
from MathExpressions import get_math_expression_from_prediction_result
from TestOneSimpleExpression import get_accuracy
from TestRotatedImages import get_bounding_boxes_for_rotated_image
from TrainRecognizeMathSymbols import prepare_data_for_prediction, run_train_model, predict, \
    prepare_data_for_prediction_lstm
from fuzzywuzzy import fuzz


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()


def read_label_from_txt_file(file_path):
    image_path = []
    labels_for_dictionary = []
    expressions = []
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            n = 2
            lines = line.split("|")
            image_path.append(lines[0])
            expressions_list = []
            expressions_list.append(lines[2])

            if len(lines) == 7:
                expressions_list.append(lines[4])
                expression = lines[6]
                expressions_list.append(expression[0:-1])
                n = 3
            else:
                expression = lines[4]
                expressions_list.append(expression[0:-1])
            expressions.append(expressions_list)
            label_list = []
            j = 1
            for index in range(0, n):
                labels = []
                symbols = lines[j].split(';')
                for symbol in symbols:
                    if len(symbol) != 0:
                        if str != "\n":
                            labels.append(symbol)
                j += 2
                label_list.append(labels)
            labels_for_dictionary.append(label_list)
            line = fp.readline()
    dictionary = dict(zip(image_path, labels_for_dictionary))
    expression_dictionary = dict(zip(image_path, expressions))

    return dictionary, expression_dictionary


def extract_expressions_from_image(image__path):
    img = cv2.imread(image__path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display_image(img)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if h < 35 and w < 70:
            thresh1[y - 30:y + h + 30, x - 30:x + w + 30] = 255
        elif w > 100 and h < 30:
            thresh1[y - 25:y + h + 25, x - 5:x + w + 5] = 255
        # else:
        #     thresh1[y - 5:y + h + 5, x - 10:x + w + 10] = 255

    # display_image(thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=13)
    # display_image(dilation)
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    im2 = img.copy()
    bounding_boxes = []
    i = 0
    while i < len(sorted_contours):
        x, y, w, h = cv2.boundingRect(sorted_contours[i])
        # print("Bounding bozxes")
        # print(w)
        # print(h)
        # print(x)
        # print(y)

        if (i + 1) < len(sorted_contours) and w < 500 and h < 400:
            bounding_boxes, im2, j = join_small_contours(sorted_contours, i, x, y, w, h, im2, bounding_boxes)
            i = j
            # display_image(im2)
        else:
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_boxes.append({
                'xmin': x,
                'xmax': x + w,
                'ymin': y,
                'ymax': y + h
            })
            # display_image(im2)
            i += 1
    # for box in bounding_boxes:
    #     print("****************")
    #     print(box['xmin'])
    #     print(box['xmax'])
    #     print(box['ymin'])
    #     print(box['ymax'])
    #     print("****************")
    # plt.subplot(1, 2, 1)
    # plt.imshow(thresh1, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(im2, cmap="gray")
    # plt.show()
    bounding_boxes = sorted(bounding_boxes, key=lambda k: (k['ymin']))
    expressions = []
    for box in bounding_boxes:
        expression = img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        expressions.append(expression)
    return expressions


def join_small_contours(sorted_contours, index, x, y, w, h, im2, bounding_boxes):
    j = index + 1
    min_x = x
    min_y = y
    max_x = x + w
    max_y = y + h
    need_to_join_contours = False
    while j < len(sorted_contours):
        x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[j])

        # print("Bounding x2")
        # print(w2)
        # print(h2)
        # print(x2)
        # print(y2)
        if (w < 300 and h < 300) and not (w2 < 300 and h2 < 300) and (j - 1) == index:
            cv2.rectangle(im2, (min(x, x2), min(y, y2)), (max(x + w, x2 + w2), max(y + h, y2 + h2)), (0, 255, 0), 2)
            bounding_boxes.append({
                'xmin': min(x, x2),
                'xmax': max(x + w, x2 + w2),
                'ymin': min(y, y2),
                'ymax': max(y + h, y2 + h2)
            })
            return bounding_boxes, im2, j + 1
        if w2 < 300 and h2 < 400:
            need_to_join_contours = True
            min_x = min(min_x, x2)
            min_y = min(min_y, y2)
            max_x = max(max_x, x2 + w2)
            max_y = max(max_y, y2 + h2)
        else:
            break

        j += 1

    if need_to_join_contours:
        cv2.rectangle(im2, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        bounding_boxes.append({
            'xmin': min_x,
            'xmax': max_x,
            'ymin': min_y,
            'ymax': max_y
        })
    else:
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes.append({
            'xmin': x,
            'xmax': x + w,
            'ymin': y,
            'ymax': y + h
        })
    return bounding_boxes, im2, j


def get_prediction_results(img, trained_model, trained_model_lstm):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


def load_rotated_image_and_return_normal_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    img = gray.copy()

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    coords = np.column_stack(np.where(thresh > 0))

    rotated_rect = cv2.minAreaRect(coords)
    (x, y), (width, height), angle = rotated_rect
    angle = cv2.minAreaRect(coords)[-1]
    # print("IMG SHAPE")
    # print(image.shape[0])
    # print(image.shape[1])
    # print("MIN AREA RECT")
    # print(height)
    # print(width)
    # print("ANGLE")
    if 1 > (image.shape[0] - image.shape[1]) > -17 and (width - height) > 0:
        angle = -angle
    elif (image.shape[0] - image.shape[1]) > 0 and (width - height) > 0:
        return image

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if 0 < (image.shape[1] - image.shape[0]) < 100 and abs(width - height) < 450 and angle > 0:
        angle = -angle
    if (abs(image.shape[1] - image.shape[0]) < 150 and abs(width - height) < 60) or (
            abs(image.shape[1] - image.shape[0]) < 30 and abs(width - height) < 150):
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

    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(rotated, cmap="gray")
    # plt.show()
    # display_image(rotated)
    return rotated


def test_several_rotated_expressions():
    predicted_labels_for_expressions = []
    truth_labels_for_expressions = []
    truth_labels_for_symbols = []
    predicted_labels_for_symbols = []
    predicted_labels_for_expressions_lstm = []
    predicted_labels_for_symbols_lstm = []

    path = 'dataset/test/several_rotated_expressions/'
    truth_symbols_dictionary, truth_expression_dictionary = read_label_from_txt_file(path + "label.txt")
    trained_model = run_train_model()
    trained_model_lstm = run_train_model_lstm()
    ratio_cnn = 0
    ratio_lstm = 0
    for img_name in os.listdir(path):
        if img_name.find("png") != -1:
            img_path = path + img_name
            expressions = extract_expressions_from_image(img_path)

            model_predicted_expressions = []
            truth_expressions = truth_expression_dictionary[img_path]
            truth_labels = truth_symbols_dictionary[img_path]
            j = 0
            model_predicted_labels = []
            for expression_img in expressions:
                # display_image(expression_img)
                img = load_rotated_image_and_return_normal_image(expression_img)
                results, expression, results_lstm, expression_lstm = get_prediction_results(img, trained_model,
                                                                                            trained_model_lstm)
                model_predicted_expressions.append(expression)
                model_predicted_labels.append(results)

                predicted_labels_for_expressions_lstm.append(expression_lstm)
                predicted_labels_for_symbols_lstm.append(results_lstm)
                ratio_cnn += fuzz.ratio(expression, truth_expressions[j])
                ratio_lstm += fuzz.ratio(expression_lstm, truth_expressions[j])

                print("TEST SEVERAL ROTATED EXPRESSIONS CNN")
                print(img_path)
                print("Thruth expression")
                print(truth_expressions[j])
                print("Predicted expression")
                print(expression)
                print("FuzzyWuzzy ratio")
                print(fuzz.ratio(expression, truth_expressions[j]))
                # print("Predicted symbols: ")
                # print(results)
                print("*****************************")

                print("TEST SEVERAL ROTATED EXPRESSIONS LSTM ")
                print(img_path)
                print("Thruth expression")
                print(truth_expressions[j])
                print("Predicted expression")
                print(expression_lstm)
                print("FuzzyWuzzy ratio")
                print(fuzz.ratio(expression_lstm, truth_expressions[j]))
                # print("Thruth symbols")
                # print(truth_labels[j])
                # print("Predicted symbols: ")
                # print(results_lstm)
                print("*****************************")

                j += 1
            if len(model_predicted_expressions) != len(truth_expressions):
                model_predicted_expressions.append('')
                new_empty_array = []
                model_predicted_labels.append(new_empty_array)
            for labels in truth_labels:
                truth_labels_for_symbols.append(labels)

            for expression in truth_expressions:
                truth_labels_for_expressions.append(expression)

            for predicted_labels in model_predicted_labels:
                predicted_labels_for_symbols.append(predicted_labels)

            for predicted_expression in model_predicted_expressions:
                predicted_labels_for_expressions.append(predicted_expression)

    print("TEST IMAGES WITH SEVERAL ROTATED EXPRESSION CNN")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols, truth_labels_for_expressions,
                 predicted_labels_for_expressions)
    print("FuzzyWazzy average ratio")
    print(round(ratio_cnn / len(truth_labels_for_expressions), 2))
    print("********************************")
    print("TEST IMAGES WITH SEVERAL ROTATED EXPRESSION LSTM ")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols_lstm, truth_labels_for_expressions,
                 predicted_labels_for_expressions_lstm)
    print("FuzzyWazzy average ratio")
    print(round(ratio_lstm / len(truth_labels_for_expressions), 2))
    print("********************************")


test_several_rotated_expressions()

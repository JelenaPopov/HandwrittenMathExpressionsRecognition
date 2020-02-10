import cv2
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import fuzz

from LSTM import predict_lstm, run_train_model_lstm
from MathExpressions import get_math_expression_from_prediction_result
from TestOneSimpleExpression import get_bounding_boxes, get_accuracy
from TrainRecognizeMathSymbols import prepare_data_for_prediction, run_train_model, predict, \
    prepare_data_for_prediction_lstm


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

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w > 100 and h < 30:
            thresh1[y - 40:y + h + 40, x:x + w + 10] = 255
        else:
            thresh1[y:y + h, x - 20:x + w + 20] = 255
    # display_image(img)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=10)
    # display_image(dilation)
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes.append({
            'xmin': x,
            'xmax': x + w,
            'ymin': y,
            'ymax': y + h
        })

    # display_image(im2)
    bounding_boxes = sorted(bounding_boxes, key=lambda k: (k['ymin']))
    new_bounding_boxes = []

    i = 0
    while i < len(bounding_boxes):
        current_box = bounding_boxes[i]
        if (i + 1) < len(bounding_boxes):
            next_box = bounding_boxes[i + 1]
            if next_box['ymin'] <= current_box['ymin'] <= next_box['ymax'] and next_box['ymin'] <= current_box[
                'ymax'] <= \
                    next_box['ymax']:
                new_bounding_boxes.append({
                    'xmin': min(current_box['xmin'], next_box['xmin']),
                    'xmax': max(current_box['xmax'], next_box['xmax']),
                    'ymin': next_box['ymin'],
                    'ymax': next_box['ymax']
                })

                i += 2
            elif current_box['ymin'] <= next_box['ymin'] <= current_box['ymax'] and current_box['ymin'] <= next_box[
                'ymax'] <= current_box['ymax']:
                new_bounding_boxes.append({
                    'xmin': min(current_box['xmin'], next_box['xmin']),
                    'xmax': max(current_box['xmax'], next_box['xmax']),
                    'ymin': current_box['ymin'],
                    'ymax': current_box['ymax']
                })

                i += 2
            else:
                new_bounding_boxes.append(current_box)
                i += 1
        else:
            new_bounding_boxes.append(current_box)
            i += 1

    sorted_boxes = sorted(new_bounding_boxes, key=lambda k: (k['ymin']))
    expressions = []
    for box in sorted_boxes:
        expression = img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        expressions.append(expression)
        # display_image(expression)
    return expressions


def get_prediction_results(img, trained_model, trained_model_lstm):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


def test_several_expressions():
    truth_labels_for_expressions = []
    truth_labels_for_symbols = []
    predicted_labels_for_symbols = []
    predicted_labels_for_expressions = []

    predicted_labels_for_expressions_lstm = []
    predicted_labels_for_symbols_lstm = []

    truth_symbols_dictionary, truth_expression_dictionary = read_label_from_txt_file(
        "dataset/test/several_expressions/label.txt")
    trained_model = run_train_model()
    trained_model_lstm = run_train_model_lstm()
    ratio_cnn = 0
    ratio_lstm = 0
    for img_name in os.listdir("dataset/test/several_expressions"):
        if img_name.find("png") != -1:
            img_path = "dataset/test/several_expressions/" + img_name
            expressions = extract_expressions_from_image(img_path)

            model_predicted_expressions = []
            truth_expressions = truth_expression_dictionary[img_path]
            truth_labels = truth_symbols_dictionary[img_path]
            j = 0
            model_predicted_labels = []
            for expression_img in expressions:
                result, expression, results_lstm, expression_lstm = get_prediction_results(expression_img,
                                                                                           trained_model,
                                                                                           trained_model_lstm)
                model_predicted_expressions.append(expression)
                model_predicted_labels.append(result)

                predicted_labels_for_expressions_lstm.append(expression_lstm)
                predicted_labels_for_symbols_lstm.append(results_lstm)
                ratio_cnn += fuzz.ratio(expression, truth_expressions[j])
                ratio_lstm += fuzz.ratio(expression_lstm, truth_expressions[j])
                print("TEST SEVERAL EXPRESSIONS CNN")
                print(img_path)
                print("Thruth expression")
                print(truth_expressions[j])
                print("Predicted expression")
                print(expression)
                print("FuzzyWuzzy ratio")
                print(fuzz.ratio(expression, truth_expressions[j]))
                # print("Thruth symbols")
                # print(truth_labels[j])
                # print("Predicted symbols: ")
                # print(result)
                print("*****************************")

                print("TEST SEVERAL EXPRESSIONS LSTM ")
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

    print("TEST IMAGES WITH SEVERAL EXPRESSIONS CNN")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols, truth_labels_for_expressions,
                 predicted_labels_for_expressions)
    print("FuzzyWazzy average ratio")
    print(round(ratio_cnn / len(truth_labels_for_expressions), 2))
    print("********************************")
    print("TEST IMAGES WITH SEVERAL EXPRESSIONS ")
    get_accuracy(truth_labels_for_symbols, predicted_labels_for_symbols_lstm, truth_labels_for_expressions,
                 predicted_labels_for_expressions_lstm)
    print("FuzzyWazzy average ratio")
    print(round(ratio_lstm / len(truth_labels_for_expressions), 2))
    print("********************************")

# test_several_expressions()

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import inkmlToPng, os


def get_traces_data(inkml_file_abs_path):
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [
                       [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                           else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(
                           float(axis_coord) * 10000) \
                                 for axis_coord in coord.split(' ')] \
                       for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                  for trace_tag in root.findall(doc_namespace + 'trace')]
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                traceDataRef = int(traceView.get('traceDataRef'))
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_data


def inkml2img(input_path, output_path):
    traces = get_traces_data(input_path)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    for elem in traces:
        ls = elem['trace_group']
        for subls in ls:
            data = np.array(subls)
            x, y = zip(*data)
            plt.plot(x, y, linewidth=2, c='black')
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.gcf().clear()


def load_inkml_and_save_to_png(test_directory):
    f = open("dataset/test/png/label.txt", "w")
    for img_name in os.listdir(test_directory):
        img_path = os.path.join(test_directory, img_name)
        new_image_path = 'dataset/test/png/' + img_name[0:-5] + "png"
        inkmlToPng.inkml2img(img_path, new_image_path)
        dictionary, annotations = get_math_expression(img_path)
        f.write(new_image_path)
        f.write(";")
        for key, value in dictionary.items():
            value = to_lower_case(value)
            f.write(value)
            f.write(";")
        f.write("|")
        f.write(annotations[0])
        f.write("\n")

    f.close()


def to_lower_case(value):
    if value == 'A':
        return 'a'
    if value == 'B':
        return 'b'
    if value == 'C':
        return 'c'
    if value == 'D':
        return 'd'
    if value == 'E':
        return 'e'
    if value == 'F':
        return 'f'
    if value == 'H':
        return 'h'
    if value == 'X':
        return 'x'
    if value == 'Y':
        return 'y'
    if value == 'gt':
        return '>'
    return value


def get_math_expression(img_path):
    trace_groups, annotations = extract_trace_groups(img_path)
    label_array = []
    id_array = []

    for trace_grp in trace_groups:
        label = trace_grp['label']
        id_traces = trace_grp['id_traces']
        if len(id_traces) > 1:
            if label[0] == "\\":
                label_array.append(label[1:])
            else:
                label_array.append(label)
            id_array.append(int(id_traces[0]))
        elif len(label) > 1:
            if label[0] == "\\":
                label_array.append(label[1:])
            else:
                label_array.append(label)
            id_array.append(int(id_traces[0]))
        else:
            label_array.append(label)
            id_array.append(int(id_traces[0]))

    dictionary = dict(zip(id_array, label_array))
    sorted_dictionary = OrderedDict(sorted(dictionary.items(), key=lambda t: t[0]))
    return sorted_dictionary, annotations


def extract_trace_groups(inkml_file_abs_path):
    trace_groups = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    traceGroupWrapper = root.findall(doc_namespace + 'traceGroup')[0]
    traceGroups = traceGroupWrapper.findall(doc_namespace + 'traceGroup')
    for traceGrp in traceGroups:
        latex_class = traceGrp.findall(doc_namespace + 'annotation')[0].text
        traceViews = traceGrp.findall(doc_namespace + 'traceView')
        id_traces = [traceView.get('traceDataRef') for traceView in traceViews]
        trace_grp = {'label': latex_class, 'traces': [], 'id_traces': id_traces}
        traces = [trace for trace in root.findall(doc_namespace + 'trace') if trace.get('id') in id_traces]
        for idx, trace in enumerate(traces):
            coords = []
            for coord in trace.text.replace('\n', '').split(','):
                coord = list(filter(None, coord.split(' ')))
                x, y = coord[:2]
                if not float(x).is_integer():
                    d_places = len(x.split('.')[-1])
                    x = float(x) * 10000
                else:
                    x = float(x)
                if not float(y).is_integer():
                    d_places = len(y.split('.')[-1])
                    y = float(y) * 10000
                else:
                    y = float(y)
                x, y = round(x), round(y)
                coords.append([x, y])
            trace_grp['traces'].append(coords)
        trace_groups.append(trace_grp)
    annotations = []
    for annotation in root.findall('{http://www.w3.org/2003/InkML}annotation'):
        if annotation.get('type') == 'truth':
            annotation_text = annotation.text
            annotation_text = annotation_text[1:]
            annotation_text = annotation_text.replace('$', '')
            annotation_text = annotation_text.replace(' ', '')
            if annotation_text.find('\\') != -1:
                annotation_text = annotation_text.replace('\\', '')
            annotations.append(annotation_text)
    return trace_groups, annotations


def read_label_from_txt_file(file_path):
    labels_for_dictionary = []
    image_path = []
    expressions = []
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            labels = []
            lines = line.split(";")
            image_path.append(lines[0])
            expression = lines[len(lines) - 1]
            expressions.append(expression[1:-1])
            lines.pop(len(lines) - 1)
            for str in lines[1:]:
                if len(str) != 0:
                    if str != "\n":
                        labels.append(str)
            labels_for_dictionary.append(labels)
            line = fp.readline()
    dictionary = dict(zip(image_path, labels_for_dictionary))
    expression_dictionary = dict(zip(image_path, expressions))
    return dictionary, expression_dictionary

def get_math_expression_from_prediction_result(results, bounding_boxes):
    index = 0
    expression = ""
    while index < len(bounding_boxes):
        express, i = check(bounding_boxes, index, results)
        expression += express
        index = i

    return expression


def check(bounding_boxes, index, results):
    if (index + 2) < len(bounding_boxes):
        fraction, i = get_fractions(bounding_boxes, results, index)
        if fraction != "":
            return fraction, i
    if (index + 1) < len(bounding_boxes):
        sqrt, i = get_sqrt(bounding_boxes, results, index)
        if sqrt != "":
            return sqrt, i
    if (index + 1) < len(bounding_boxes):
        superscripts, i = get_superscripts(bounding_boxes, results, index)
        if superscripts != "":
            return superscripts, i
        subscripts, i = get_subscripts(bounding_boxes, results, index)
        if subscripts != "":
            return subscripts, i
    expression = results[index]
    index += 1
    return expression, index


def get_fractions(bounding_boxes, results, index):
    i = index + 1
    box = bounding_boxes[index]
    counter = ""
    denominator = ""
    if results[index] == "-":
        while i < len(bounding_boxes):
            next_box = bounding_boxes[i]
            if box['xmin'] <= next_box['xmin'] <= box['xmax'] and box['xmin'] <= next_box['xmax'] <= \
                    box['xmax']:
                if next_box['ymin'] <= box['ymin']:
                    express, j = check(bounding_boxes, i, results)
                    denominator += express
                    i = j
                elif next_box['ymin'] + 2 >= box['ymin']:
                    express, j = check(bounding_boxes, i, results)
                    counter += express
                    i = j
                else:
                    break
            else:
                break

        if len(counter) > 0 and len(denominator) > 0:
            return "frac{" + denominator + "}{" + counter + "}", i
    return "", index


def get_superscripts(bounding_boxes, results, index):
    i = index + 1
    box = bounding_boxes[index]
    expression = results[index]
    superscripts = ""
    box_height = box['ymax'] - box['ymin']
    bound = box['ymin'] + (1 / 2) * box_height
    can_not_have_superscripts = [
        '-', '!', '(', ',', '[', '{', '+', '=', 'cos', 'div', 'exists', 'forall', 'forward_slash', 'geq',
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'log', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'sin', 'sqrt', 'sum', 'tan', 'times',
    ]
    if any(elem == results[index] for elem in can_not_have_superscripts):
        return "", index
    can_not_be_superscript = [
        '!', ',', '=', 'div', 'exists', 'forall', 'forward_slash', 'geq',
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'cos', 'sin', 'sum', 'tan', 'times',
    ]
    while i < len(bounding_boxes):
        if any(elem == results[i] for elem in can_not_be_superscript):
            break
        if bound >= bounding_boxes[i]['ymin'] and bounding_boxes[i]['ymax'] - 5 < bound and bounding_boxes[i]['xmin'] - \
                bounding_boxes[i - 1]['xmax'] < 100 and bounding_boxes[i]['ymin'] < box['ymin']:
            if box['xmin'] <= bounding_boxes[i]['xmin'] < box['xmax'] and box['xmin'] <= bounding_boxes[i]['xmax'] < \
                    box['xmax']:
                break
            express, j = check(bounding_boxes, i, results)
            superscripts += express
            i = j
        elif results[index] == ")" and bound >= bounding_boxes[i]['ymin'] and bounding_boxes[i]['ymax'] < bound and \
                bounding_boxes[i]['xmin'] - \
                bounding_boxes[i - 1]['xmax'] < 100:
            express, j = check(bounding_boxes, i, results)
            superscripts += express
            i = j
        else:
            break
    can_not_be_superscript = [
        '-', '!', '(', ',', '[', '{', '+', '=', 'cos', 'div', 'exists', 'forall', 'forward_slash', 'geq',
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'log', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'sin', 'sum', 'tan', 'times',
    ]
    if superscripts != "":
        if len(superscripts) == 1:
            if any(elem == superscripts for elem in can_not_be_superscript):
                return "", index
            expression += "^" + superscripts
        else:
            if superscripts.find("+") == 0:
                return "", index
            if any(elem == superscripts[-1] for elem in can_not_be_superscript):
                superscripts = superscripts[:-1]
                i -= 1
            expression += "^{" + superscripts + "}"
        return expression, i
    return "", index


def get_subscripts(bounding_boxes, results, index):
    i = index + 1
    box = bounding_boxes[index]
    expression = results[index]
    subscripts = ""
    box_height = box['ymax'] - box['ymin']
    bound = box['ymax'] - (1 / 2) * box_height
    can_not_have_subscripts = [
        '-', '!', '(', ')', ',', '[', ']', '{', '}', '+', '=', '0', '1', '2',
        '3', '4', '5', '6', '7', '8', '9', 'cos', 'div', 'exists', 'forall', 'forward_slash', 'geq', 'gt',
        'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'sin', 'sqrt', 'sum', 'tan', 'times',
    ]
    if any(elem == results[index] for elem in can_not_have_subscripts):
        return "", index
    can_not_be_subscript = [
        '!', ',', '=', 'div', 'exists', 'forall', 'forward_slash', 'geq',
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'cos', 'sin', 'sum', 'tan', 'times',
    ]
    while i < len(bounding_boxes):
        if any(elem == results[i] for elem in can_not_be_subscript):
            break
        next_box = bounding_boxes[i]
        next_box_height = (next_box['ymax'] - next_box['ymin']) * (1 / 2)
        if bound <= next_box['ymin'] and next_box['xmin'] - bounding_boxes[i - 1]['xmax'] <= 100 and box[
            'ymax'] >= next_box['ymin'] >= box['ymin'] and box['ymax'] + 5 <= next_box['ymax']:
            if results[index] == "log":
                subscripts += results[i]
                i += 1
                break
            express, j = check(bounding_boxes, i, results)
            subscripts += express
            i = j
        elif results[index] == "log" and next_box['ymax'] > bound and next_box['ymin'] >= bound:
            subscripts += results[i]
            i += 1
            break
        elif (next_box['ymax'] - bound) >= next_box_height and results[i] != "-" and results[i] != "+" and next_box[
            'ymax'] <= box['ymax']:
            express, j = check(bounding_boxes, i, results)
            subscripts += express
            i = j
        else:
            break
    can_not_be_alone_in_subscripts = [
        '-', '!', '(', ',', '[', '{', '+', '=', 'cos', 'div', 'exists', 'forall', 'forward_slash', 'geq',
        'gt', 'in', 'infty', 'int', 'ldots', 'leq', 'lim', 'lt', '<',
        'neq', 'phi', 'pi', 'pm', 'rightarrow', 'sin', 'sum', 'tan', 'times',
    ]
    if subscripts != "":
        if len(subscripts) == 1:
            if any(elem == subscripts for elem in can_not_be_alone_in_subscripts):
                return "", index
            if subscripts != "+" or subscripts != "-":
                if results[index] == "log":
                    expression += "_{" + subscripts + "}"
                else:
                    expression += "_" + subscripts
            else:
                return "", index
        else:
            if subscripts.find("=") != -1:
                return "", index
            if any(elem == subscripts[-1] for elem in can_not_be_alone_in_subscripts):
                subscripts = subscripts[:-1]
            expression += "_{" + subscripts + "}"
        return expression, i
    return "", index


def get_sqrt(bounding_boxes, results, index):
    i = index + 1
    box = bounding_boxes[index]
    in_sqrt = ""
    if results[index] == "sqrt":
        while i < len(bounding_boxes):
            next_box = bounding_boxes[i]
            if box['xmin'] <= next_box['xmin'] <= box['xmax'] and box['xmin'] <= next_box['xmax'] <= \
                    box['xmax']:
                express, j = check(bounding_boxes, i, results)
                in_sqrt += express
                i = j
            else:
                break

        if len(in_sqrt) > 0:
            return "sqrt{" + in_sqrt + "}", i
    return "", index

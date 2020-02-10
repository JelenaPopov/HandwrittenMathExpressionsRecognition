from random import seed, randint
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import inkmlToPng


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def make_rotated_images(test_directory):
    j = 0
    path = 'dataset/test/rotated_image/'
    f = open("dataset/test/rotated_image/label.txt", "w")
    seed(1)
    for img_name in os.listdir(test_directory):
        if j == 100:
            break
        if img_name.find("png") != -1:
            rotation_angle = randint(-45, 44)
            print(rotation_angle)
            img_path = os.path.join(test_directory, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_inkml_path = os.path.join("dataset/test/one_expression", img_name[0:-3] + "inkml")
            dictionary, annotations = inkmlToPng.get_math_expression(img_inkml_path)
            rotated = rotate_bound(img, rotation_angle)
            cv2.imwrite(os.path.join(path, img_name), rotated)
            save_label_in_txt(f, path + img_name, dictionary, annotations)
            cv2.waitKey(0)
            j += 1
    f.close()


def save_label_in_txt(f, image_path, dictionary, annotations):
    f.write(image_path)
    f.write(";")
    for key, value in dictionary.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
    f.write("|")
    f.write(annotations[0])
    f.write("\n")


def get_concat_v_blank(im1, im2, x_offset_1, x_offset_2, y_offset):
    dst = Image.new('RGB', (max(im1.width, im2.width) + 100, im1.height + im2.height + 100), (255, 255, 255))
    dst.paste(im1, (x_offset_1, 0))
    dst.paste(im2, (x_offset_2, im1.height + y_offset))
    return dst


def load_image(test_directory, img_name):
    img_path = os.path.join(test_directory, img_name)
    return Image.open(img_path)


def get_image_annotations(img_name):
    img_inkml_path = os.path.join("dataset/test/one_expression", img_name[0:-3] + "inkml")
    return inkmlToPng.get_math_expression(img_inkml_path)


def make_images_with_several_expressions(test_directory):
    path = 'dataset/test/several_expressions/'
    f = open("dataset/test/several_expressions/label.txt", "w")
    seed(1)
    images_name = os.listdir(test_directory)
    index = 0
    index2 = 0
    i = 0
    j = 1
    z = 2
    while z < len(images_name):
        # make image with two expression
        if i <= 100:
            img_name_1 = images_name[i]
            img_name_2 = images_name[j]
            if img_name_1.find("png") != -1 and img_name_2.find("png") != -1:
                img_1 = load_image(test_directory, img_name_1)
                img_2 = load_image(test_directory, img_name_2)
                new_image = get_concat_v_blank(img_1, img_2, randint(10, 50), randint(10, 50), randint(20, 80))
                new_image_path = path + 'two_expression_in_one_image' + str(index) + '.png'
                new_image.save(new_image_path)
                save_label_for_two_expressions_in_txt(f, new_image_path, img_name_1,
                                                      img_name_2)
                cv2.waitKey(0)
                index += 1

            i += 2
            j += 2
            z += 2
        else:
            # make image with three expressions
            img_name_1 = images_name[i]
            img_name_2 = images_name[j]
            img_name_3 = images_name[z]
            if img_name_1.find("png") != -1 and img_name_2.find("png") != -1 and img_name_3.find("png") != -1:
                img_1 = load_image(test_directory, img_name_1)
                img_2 = load_image(test_directory, img_name_2)
                img_3 = load_image(test_directory, img_name_3)
                new_image_1 = get_concat_v_blank(img_1, img_2, randint(10, 50), randint(10, 50), randint(20, 80))
                new_image = get_concat_v_blank(new_image_1, img_3, randint(10, 50), randint(0, 50), randint(20, 80))
                new_image_path = path + 'three_expression_in_one_image' + str(index2) + '.png'
                new_image.save(new_image_path)
                save_label_for_three_expressions_in_txt(f, new_image_path, img_name_1, img_name_2, img_name_3)
                cv2.waitKey(0)
                index2 += 1

            i += 3
            j += 3
            z += 3
    f.close()


def save_label_for_two_expressions_in_txt(f, image_path, img_name_1, img_name_2):
    dictionary_1, annotations_1 = get_image_annotations(img_name_1)
    dictionary_2, annotations_2 = get_image_annotations(img_name_2)
    f.write(image_path)
    f.write("|")
    for key, value in dictionary_1.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
    f.write("|")
    f.write(annotations_1[0])
    f.write("|")
    for key, value in dictionary_2.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
    f.write("|")
    f.write(annotations_2[0])

    f.write("\n")


def save_label_for_three_expressions_in_txt(f, image_path, img_name_1, img_name_2, img_name_3):
    dictionary_1, annotations_1 = get_image_annotations(img_name_1)
    dictionary_2, annotations_2 = get_image_annotations(img_name_2)
    dictionary_3, annotations_3 = get_image_annotations(img_name_3)
    f.write(image_path)
    f.write("|")
    for key, value in dictionary_1.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
    f.write("|")
    f.write(annotations_1[0])
    f.write("|")
    for key, value in dictionary_2.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
        f.write(";")
    f.write("|")
    f.write(annotations_2[0])
    f.write("|")
    for key, value in dictionary_3.items():
        value = inkmlToPng.to_lower_case(value)
        f.write(value)
        f.write(";")
    f.write("|")
    f.write(annotations_3[0])
    f.write("\n")


def make_rotated_images_for_creating_image_with_several_expressions(test_directory):
    path = 'dataset/test/several_rotated_expressions/rotated_image/'
    seed(1)
    for img_name in os.listdir(test_directory):
        if img_name.find("png") != -1:
            rotation_angle = randint(-45, 44)
            print(rotation_angle)
            img_path = os.path.join(test_directory, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rotated = rotate_bound(img, rotation_angle)
            cv2.imwrite(os.path.join(path, img_name), rotated)


def make_images_with_several_rotated_expressions(test_directory):
    make_rotated_images_for_creating_image_with_several_expressions(test_directory)
    path = 'dataset/test/several_rotated_expressions/'
    f = open("dataset/test/several_rotated_expressions/label.txt", "w")
    seed(1)
    images_name = os.listdir('dataset/test/several_rotated_expressions/rotated_image/')
    index = 0
    index2 = 0
    i = 0
    j = 1
    z = 2
    while z < len(images_name):
        # make image with two expression
        if i <= 100:
            img_name_1 = images_name[i]
            img_name_2 = images_name[j]
            if img_name_1.find("png") != -1 and img_name_2.find("png") != -1:
                img_1 = load_image('dataset/test/several_rotated_expressions/rotated_image/', img_name_1)
                img_2 = load_image('dataset/test/several_rotated_expressions/rotated_image/', img_name_2)
                new_image = get_concat_v_blank(img_1, img_2, randint(10, 50), randint(10, 50),
                                               randint(20, 80))
                new_image_path = path + 'two_expression_in_one_image' + str(index) + '.png'
                new_image.save(new_image_path)
                save_label_for_two_expressions_in_txt(f, new_image_path, img_name_1,
                                                      img_name_2)
                cv2.waitKey(0)
                index += 1

            i += 2
            j += 2
            z += 2
        else:
            # make image with three expressions
            img_name_1 = images_name[i]
            img_name_2 = images_name[j]
            img_name_3 = images_name[z]
            if img_name_1.find("png") != -1 and img_name_2.find("png") != -1 and img_name_3.find("png") != -1:
                img_1 = load_image('dataset/test/several_rotated_expressions/rotated_image/', img_name_1)
                img_2 = load_image('dataset/test/several_rotated_expressions/rotated_image/', img_name_2)
                img_3 = load_image('dataset/test/several_rotated_expressions/rotated_image/', img_name_3)
                new_image_1 = get_concat_v_blank(img_1, img_2, randint(10, 50), randint(10, 50),
                                                 randint(20, 80))
                new_image = get_concat_v_blank(new_image_1, img_3, randint(10, 50), randint(10, 50),
                                               randint(20, 80))
                new_image_path = path + 'three_expression_in_one_image' + str(index2) + '.png'
                new_image.save(new_image_path)
                save_label_for_three_expressions_in_txt(f, new_image_path, img_name_1, img_name_2, img_name_3)
                cv2.waitKey(0)
                index2 += 1

            i += 3
            j += 3
            z += 3
    f.close()

# make_rotated_images("dataset/test/png/")
# make_images_with_several_expressions("dataset/test/png/")
# make_images_with_several_rotated_expressions("dataset/test/png/")

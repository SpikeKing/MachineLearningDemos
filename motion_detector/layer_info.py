# -- coding: utf-8 --
"""

created by C.L.Wang
"""
import os

from keras import models
from keras.models import load_model

from motion_detector.pic_drawing import draw_dict
from project_utils import read_file
from root_dir import ROOT_DIR
import numpy as np
from motion_detector.merged_dcl import load_data


def show_layer():
    """
    绘制层次的信号图
    :return: 写入图片
    """
    data_path = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")
    output_path = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")

    lm_path = os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset/activity_labels.txt")
    lm_lines = read_file(lm_path)

    label_meaning_dict = dict()
    for lm in lm_lines:
        [label, meaning] = lm.split(" ")
        label_meaning_dict[label] = meaning.rstrip()

    label_path = os.path.join(data_path, "train", "y_train.txt")

    labels = np.loadtxt(label_path)
    label_index_dict = dict()

    for label in label_meaning_dict.keys():
        data_index = np.where(labels == int(label))[0][0]
        meaning = label_meaning_dict[str(label)]
        label_index_dict[meaning] = data_index

    print "label_index_dict: %s" % label_index_dict

    model_path = os.path.join(output_path, "merged_dcl.h5")
    model = load_model(model_path)
    model.summary()  # As a reminder.
    # 48是Merge层, 3是卷积层
    activation_model = models.Model(inputs=model.input, outputs=[model.layers[48].output])
    X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val = load_data(data_path)

    label_data_dict = dict()
    for label in label_index_dict.keys():
        index = label_index_dict[label]
        activations = activation_model.predict(
            [np.expand_dims(X_trainS1[index], axis=0), np.expand_dims(X_trainS2[index], axis=0),
             np.expand_dims(X_trainS3[index], axis=0)])

        label_data_dict[label] = activations[0]

    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'black']
    out_path = os.path.join(ROOT_DIR, output_path, "merged_layer.png")
    draw_dict(label_data_dict, color_list, out_path)


if __name__ == '__main__':
    show_layer()

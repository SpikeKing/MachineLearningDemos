# -- coding: utf-8 --
"""
绘制传感器数据

created by C.L.Wang
"""
import matplotlib.pyplot as plt
from cycler import cycler

from project_utils import *
from root_dir import ROOT_DIR


def show_data_of_motions():
    """
    显示不同运动的曲线
    :return: 曲线图
    """
    lm_path = os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset/activity_labels.txt")
    lm_lines = read_file(lm_path)

    label_meaning_dict = dict()
    for lm in lm_lines:
        [label, meaning] = lm.split(" ")
        label_meaning_dict[label] = meaning.rstrip()

    train_dir = os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset/train")
    data_path = os.path.join(train_dir, "Inertial Signals/body_acc_x_train.txt")
    label_path = os.path.join(train_dir, "y_train.txt")

    data = np.loadtxt(data_path)
    labels = np.loadtxt(label_path)
    label_data_dict = dict()

    for label in label_meaning_dict.keys():
        data_index = np.where(labels == int(label))[0][0]
        meaning = label_meaning_dict[str(label)]
        label_data_dict[meaning] = data[data_index]

    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'black']
    out_path = os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset_output/activity_lines.png")
    draw_dict(label_data_dict, color_list, out_path)


def show_data_of_sensors():
    """
    显示不同传感器的曲线
    :return: 曲线图
    """
    paths, names = listdir_files(os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset/train/Inertial Signals"))
    label_data_dict = dict()
    for path, name in zip(paths, names):
        data = np.loadtxt(path)[27]
        label = name.rstrip('.txt')
        label_data_dict[label] = data
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'hotpink', 'brown', 'crimson']
    out_path = os.path.join(ROOT_DIR, "data/UCI_HAR_Dataset_output/sensors_lines.png")
    draw_dict(label_data_dict, color_list, out_path)


def draw_dict(label_data_dict, color_list, out_path):
    """
    绘制字典中的数据
    :param label_data_dict: 数据字典
    :param color_list: 颜色列表
    :param out_path: 输出文件
    :return: None
    """
    label_data_list = sorted(label_data_dict.items())  # 字典排序

    plt.gca().set_prop_cycle(cycler('color', color_list))  # 图片颜色
    plt.gcf().set_size_inches(18.5, 10.5)  # 图片尺寸

    label_list = []
    for label, data in label_data_list:
        plt.plot(data)  # 写入绘制数据
        label_list.append(label)

    plt.legend(label_list, loc='upper left')  # 字典标签
    plt.grid()  # 网格
    plt.savefig(out_path)  # 写入图片
    plt.show()


if __name__ == '__main__':
    show_data_of_motions()
    show_data_of_sensors()

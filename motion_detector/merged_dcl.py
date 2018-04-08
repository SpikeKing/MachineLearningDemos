# -- coding: utf-8 --

import os

import sys
from keras import Input, Model
from keras.callbacks import Callback
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense, Activation, np, Concatenate
from keras.utils import to_categorical, plot_model
from sklearn.metrics import precision_recall_fscore_support

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from project_utils import create_folder_try
from root_dir import ROOT_DIR


def load_data(data_path):
    """
    加载本地的UCI的训练数据和验证数据
    :param data_path 数据集
    :return: 训练数据和验证数据
    """
    train_path = os.path.join(data_path, "train")
    train_X_path = os.path.join(train_path, "Inertial Signals")

    X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt"))
    X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt"))
    X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt"))
    X_trainS1 = np.array([X_trainS1_x, X_trainS1_y, X_trainS1_z])
    X_trainS1 = X_trainS1.transpose([1, 2, 0])

    X_trainS2_x = np.loadtxt(os.path.join(train_X_path, "body_gyro_x_train.txt"))
    X_trainS2_y = np.loadtxt(os.path.join(train_X_path, "body_gyro_y_train.txt"))
    X_trainS2_z = np.loadtxt(os.path.join(train_X_path, "body_gyro_z_train.txt"))
    X_trainS2 = np.array([X_trainS2_x, X_trainS2_y, X_trainS2_z])
    X_trainS2 = X_trainS2.transpose([1, 2, 0])

    X_trainS3_x = np.loadtxt(os.path.join(train_X_path, "total_acc_x_train.txt"))
    X_trainS3_y = np.loadtxt(os.path.join(train_X_path, "total_acc_y_train.txt"))
    X_trainS3_z = np.loadtxt(os.path.join(train_X_path, "total_acc_z_train.txt"))
    X_trainS3 = np.array([X_trainS3_x, X_trainS3_y, X_trainS3_z])
    X_trainS3 = X_trainS3.transpose([1, 2, 0])

    Y_train = np.loadtxt(os.path.join(train_path, "y_train.txt"))
    Y_train = to_categorical(Y_train - 1.0)  # 标签是从1开始

    print "训练数据: "
    print "传感器1: %s, 传感器1的X轴: %s" % (str(X_trainS1.shape), str(X_trainS1_x.shape))
    print "传感器2: %s, 传感器2的X轴: %s" % (str(X_trainS2.shape), str(X_trainS2_x.shape))
    print "传感器3: %s, 传感器3的X轴: %s" % (str(X_trainS3.shape), str(X_trainS3_x.shape))
    print "传感器标签: %s" % str(Y_train.shape)
    print ""

    test_path = os.path.join(data_path, "test")
    test_X_path = os.path.join(test_path, "Inertial Signals")

    X_valS1_x = np.loadtxt(os.path.join(test_X_path, "body_acc_x_test.txt"))
    X_valS1_y = np.loadtxt(os.path.join(test_X_path, "body_acc_y_test.txt"))
    X_valS1_z = np.loadtxt(os.path.join(test_X_path, "body_acc_z_test.txt"))
    X_valS1 = np.array([X_valS1_x, X_valS1_y, X_valS1_z])
    X_valS1 = X_valS1.transpose([1, 2, 0])

    X_valS2_x = np.loadtxt(os.path.join(test_X_path, "body_gyro_x_test.txt"))
    X_valS2_y = np.loadtxt(os.path.join(test_X_path, "body_gyro_y_test.txt"))
    X_valS2_z = np.loadtxt(os.path.join(test_X_path, "body_gyro_z_test.txt"))
    X_valS2 = np.array([X_valS2_x, X_valS2_y, X_valS2_z])
    X_valS2 = X_valS2.transpose([1, 2, 0])

    X_valS3_x = np.loadtxt(os.path.join(test_X_path, "total_acc_x_test.txt"))
    X_valS3_y = np.loadtxt(os.path.join(test_X_path, "total_acc_y_test.txt"))
    X_valS3_z = np.loadtxt(os.path.join(test_X_path, "total_acc_z_test.txt"))
    X_valS3 = np.array([X_valS3_x, X_valS3_y, X_valS3_z])
    X_valS3 = X_valS3.transpose([1, 2, 0])

    Y_val = np.loadtxt(os.path.join(test_path, "y_test.txt"))
    Y_val = to_categorical(Y_val - 1.0)

    print "验证数据: "
    print "传感器1: %s, 传感器1的X轴: %s" % (str(X_valS1.shape), str(X_valS1.shape))
    print "传感器2: %s, 传感器2的X轴: %s" % (str(X_valS2.shape), str(X_valS2.shape))
    print "传感器3: %s, 传感器3的X轴: %s" % (str(X_valS3.shape), str(X_valS3.shape))
    print "传感器标签: %s" % str(Y_val.shape)
    print "\n"

    return X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val


class Metrics(Callback):
    """
    输出F, P, R
    """

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, batch, logs=None):
        val_targ = self.validation_data[-3]

        val_value = [x for x in self.validation_data[0:-3]]
        y_pred = np.asarray(self.model.predict(val_value))

        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_targ, (y_pred > 0.5).astype(int), average='micro')
        print "— val_f1: % f — val_precision: % f — val_recall % f" % (f_score, precision, recall)


def main(data_path, output_path):
    X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val = load_data(data_path)

    epochs = 100
    batch_size = 256
    kernel_size = 3
    pool_size = 2
    dropout_rate = 0.15
    f_act = 'relu'
    n_classes = 6

    # 三个子模型的输入数据
    main_input1 = Input(shape=(128, 3), name='main_input1')
    main_input2 = Input(shape=(128, 3), name='main_input2')
    main_input3 = Input(shape=(128, 3), name='main_input3')

    def cnn_lstm_cell(main_input):
        """
        基于DeepConvLSTM算法, 创建子模型
        :param main_input: 输入数据
        :return: 子模型
        """
        sub_model = Conv1D(512, kernel_size, input_shape=(128, 3), activation=f_act, padding='same')(main_input)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(64, kernel_size, activation=f_act, padding='same')(sub_model)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(32, kernel_size, activation=f_act, padding='same')(sub_model)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        sub_model = LSTM(128, return_sequences=True)(sub_model)
        sub_model = LSTM(128, return_sequences=True)(sub_model)
        sub_model = LSTM(128)(sub_model)
        main_output = Dropout(dropout_rate)(sub_model)
        return main_output

    first_model = cnn_lstm_cell(main_input1)
    second_model = cnn_lstm_cell(main_input2)
    third_model = cnn_lstm_cell(main_input3)

    model = Concatenate()([first_model, second_model, third_model])  # 合并模型
    model = Dropout(0.4)(model)
    model = Dense(n_classes)(model)
    model = BatchNormalization()(model)
    output = Activation('softmax', name="softmax")(model)

    model = Model([main_input1, main_input2, main_input3], output)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    graph_path = os.path.join(output_path, "merged_model.png")
    plot_model(model, to_file=graph_path, show_shapes=True)  # 绘制模型图

    metrics = Metrics()  # 度量FPR
    history = model.fit([X_trainS1, X_trainS2, X_trainS3], Y_train,
                        batch_size=batch_size,
                        validation_data=([X_valS1, X_valS2, X_valS3], Y_val),
                        epochs=epochs,
                        callbacks=[metrics])  # 增加FPR输出

    model_path = os.path.join(output_path, "merged_dcl.h5")
    model.save(model_path)  # 存储模型
    print history.history


if __name__ == '__main__':
    data = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")
    output = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")
    create_folder_try(output)
    main(data_path=data, output_path=output)

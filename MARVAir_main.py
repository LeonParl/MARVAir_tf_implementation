## system related packages
import os
import os.path as op
import shutil
import gc
from multiprocessing import Pool

## functional packages
import numpy as np
import pandas as pd

import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.metrics as mc

import time

from functools import partial

import tensorflow as tf

from tensorflow.keras import initializers

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv1D, Multiply, Add, GlobalAveragePooling1D
from tensorflow.keras.layers import MaxPooling1D, Input

from tensorflow.keras.applications import ResNet50V2

## toggle if you wish to disable CUDA.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

## to ignore some legacy errors produced by numpy
np.warnings.filterwarnings('error', category = np.VisibleDeprecationWarning)


class DataProcessor:
    def __init__(self, server_val):
        print('initiate DataProcessor')

        self.root_dir = op.abspath(os.sep)

        self.extend_load_vis = 8
        self.extend_load_met = 12800
        self.limited_load_vis = 2
        self.limited_load_met = 1600

        if server_val == 'local':
            self.batch_size_vis = self.extend_load_vis
            self.batch_size_met = self.extend_load_met
            # raise NotImplementedError('Actually this directory has not been defined. ')
            self.image_assets_dir = str('/home/exp/data/AQI_Data/hyb_3d_rad_corr')
            self.desc_file_dir = str(r'/home/exp/data/AQI_Data/')
        else:
            self.batch_size_vis = self.limited_load_vis
            self.batch_size_met = self.limited_load_met
            self.image_assets_dir = str(r'./hyb_3d_rad_corr')
            self.desc_file_dir = str(r'./')

        ## define the num of work load
        self.worker_num = 20

        self.desc_path = op.join(self.desc_file_dir + '/AQI_label_c3.csv')

        print('the data being used is under', self.image_assets_dir)
        np.random.seed(13)
        tf.random.set_seed(1313)

        self.init_fusion_data_loading()

    def __del__(self):
        print('the instance of DataProcessor has been recycled. ')

    def init_fusion_data_loading(self):
        ## in this function, the data used to train the both sub-models will be loaded here.

        ## loading the dataset description
        data = pd.read_csv(self.desc_path)

        ## getting the target label ready
        label_seq = data.loc[:, 'proc_aqi'].to_numpy().reshape(-1, )

        ## one-hot encoder for label
        ohenc_label = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        ohenc_label.fit(label_seq.reshape(-1, 1))

        ## counting unnecessary columns to be removed in the dataset description
        remove_col_list = ['id', 'model', 'timestamp', 'pm25', 'date', 'proc_aqi']
        remain_col_list = [ele for ele in data.columns.to_list() if ele not in remove_col_list]

        ## counting columns to be one-hot encoded in the dataset description
        oh_col_list = ['location', 'type', 'Weather', 'year', 'month', 'day', 'weekday']

        ## counting columns to be max-min scaled in the dataset description
        remove_col_list_sc_add = ['proc_tar']
        remove_col_list_sc = [*remove_col_list, *oh_col_list, *remove_col_list_sc_add]
        remain_col_list_sc = [ele for ele in data.columns.to_list() if ele not in remove_col_list_sc]

        ## performing the max-min scaling and one-hot encoding
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(data[remain_col_list_sc].values)
        data.loc[:, remain_col_list_sc] = mm_scaler.transform(data[remain_col_list_sc].values)

        ohenc_content = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        onehot_output = ohenc_content.fit_transform(data[oh_col_list].values).astype(int)

        proc_col_list = [ele for ele in remain_col_list if ele not in oh_col_list]
        concat_col = proc_col_list + list('oh_' + str(i) for i in np.arange(onehot_output.shape[1]))
        data_proc = pd.DataFrame(np.hstack((data[proc_col_list].to_numpy(), onehot_output)), columns = concat_col)

        ## splitting the dataset description into two parts
        train_part, valid_part, label_train, label_valid = train_test_split(data_proc, label_seq,
                                                                            test_size = 0.2, shuffle = True)

        ## doing one_hot encoding for label
        self.label_train_raw = label_train
        self.label_valid_raw = label_valid
        self.label_train = ohenc_label.transform(label_train.reshape(-1, 1))
        self.label_valid = ohenc_label.transform(label_valid.reshape(-1, 1))

        self.shape_val = self.label_valid.shape

        ## splitting the description part
        self.train_img_seq = np.array([self.image_assets_dir + str('/') + str(i)
                                       for i in train_part['proc_tar']]).reshape(-1, )
        self.valid_img_seq = np.array([self.image_assets_dir + str('/') + str(i)
                                       for i in valid_part['proc_tar']]).reshape(-1, )

        meteor_col = concat_col
        meteor_col.remove('proc_tar')
        self.train_met_part = self.meteor_part_shaper(train_part[meteor_col], 1, 1)
        self.valid_met_part = self.meteor_part_shaper(valid_part[meteor_col], 1, 1)
        self.met_part_shape = self.valid_met_part.shape

        ## utilizes multiprocessing for data loading here.
        self.tar_img_row = 448
        self.tar_img_col = 448
        timer_start = time.perf_counter()

        adv_arg_setter = partial(adv_img_loader, self.tar_img_col, self.tar_img_row)
        train_img_loader_pool = Pool(processes = self.worker_num)
        print('preparing the training set. ')
        train_vis_part = train_img_loader_pool.map(adv_arg_setter, self.train_img_seq)
        train_img_loader_pool.close()
        train_img_loader_pool.join()
        self.train_vis_part = np.array(train_vis_part)
        del train_vis_part, train_img_loader_pool
        gc.collect()

        ## please note that the previous pool has been completely shut down after
        ## so the instance of that pool cannot be used twice.
        valid_img_loader_pool = Pool(processes = self.worker_num)
        print('preparing the validation set. ')
        adv_arg_setter = partial(adv_img_loader, self.tar_img_col, self.tar_img_row)
        valid_vis_part = valid_img_loader_pool.map(adv_arg_setter, self.valid_img_seq)
        valid_img_loader_pool.close()
        valid_img_loader_pool.join()
        self.valid_vis_part = np.array(valid_vis_part)
        del valid_vis_part, valid_img_loader_pool
        gc.collect()

        elapsed_time = round(((time.perf_counter() - timer_start) / 60), 3)
        print('visual assets preparation finished. elapsed time:', elapsed_time, 'mins. ')

        # print('test')
        # pass
        del train_part, valid_part, label_train, label_valid
        del data, mm_scaler, ohenc_content, ohenc_label
        gc.collect()

    def meteor_part_shaper(self, df_con, time_steps, stride):
        feature = []
        for i in range(0, len(df_con), stride):
            temp = df_con.iloc[i:(i + time_steps)].values
            feature.append(temp)
        return np.array(feature).astype(np.float64)


def adv_img_loader(tar_img_col, tar_img_row, tar_img_path):
    img = cv2.imread(tar_img_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (tar_img_col, tar_img_row))
    return resized_img


class BuildVisualCore:
    def __init__(self):
        print('now building the visual core. ')
        self.current_epoch = int(1)
        self.acc_epoch = int(0)

        self.init_visual_core()

    def __del__(self):
        print('the instance of visual core has been recycled. ')

    def init_visual_core(self):
        base_model = ResNet50V2(include_top = False, weights = 'imagenet',
                                input_shape = (448, 448, 3), pooling = 'avg')
        base_model._name = 'ResNet_Backbone'
        base_model.layers[0]._name = 'visual_input'
        base_model.summary()

        ## adding regression layers to the backbone
        fc1 = Dense(256, activation = 'elu', name = 'fc1')(base_model.get_layer('avg_pool').output)
        fc2 = Dense(16, activation = 'elu', name = 'fc2')(fc1)
        inf_output = Dense(dp.shape_val[1], activation = 'softmax', name = 'vis_inference')(fc2)

        visual_core = Model(inputs = base_model.input,
                            outputs = inf_output)
        visual_core._name = 'Visual_Core'

        layer_cond = ('conv1', 'conv2')
        for layer in visual_core.layers:
            if layer._name.startswith(layer_cond):
                layer.trainable = False

        visual_core.summary()
        # print('test')
        visual_core.compile(loss = 'categorical_crossentropy',
                            optimizer = 'adam',
                            metrics = ['accuracy',
                                       tf.keras.metrics.Precision(),
                                       tf.keras.metrics.Recall()])

        self.visual_core = visual_core

    def load_tuned_model(self):
        eva_model = load_model('./visual_' + str(self.acc_epoch) + '/model/visual_core.h5')
        eva_model.summary()

        self.eva_model = eva_model


class BuildMeteorCore:
    def __init__(self):
        print('initiating the settings of the model. ')

        self.hybrid_meteor_model()

        self.current_epoch = int(1)
        self.acc_epoch = int(0)

    def __del__(self):
        print('the instance of meteor core has been recycled. ')

    def hybrid_meteor_model(self):
        ## try to add residual-alike structure here.

        init = initializers.glorot_uniform(seed = 13)
        data_shape_val = dp.met_part_shape

        input_meteor = Input(shape = (data_shape_val[1], data_shape_val[2]), name = 'meteor_input')
        att_k1 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_1')(input_meteor)
        att_k2 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_2')(input_meteor)
        att_k3 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_3')(input_meteor)
        att_k4 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_4')(input_meteor)
        att_k5 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_5')(input_meteor)
        att_k6 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_6')(input_meteor)
        att_k7 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_7')(input_meteor)
        att_k8 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_8')(input_meteor)
        att_k9 = Conv1D(32, 1, activation = 'elu', kernel_initializer = init, padding = 'same',
                        name = 'att_9')(input_meteor)

        hybrid_input = tf.keras.layers.concatenate([att_k1, att_k2, att_k3,
                                                    att_k4, att_k5, att_k6,
                                                    att_k7, att_k8, att_k9], 1, name = 'met_concat')
        hybrid_do = Dropout(0.25, name = 'met_dropout')(hybrid_input)

        ## conv1d block 2
        c2d_1 = Dense(64, kernel_initializer = init, activation = 'elu', name = 'met_c2d_1')(hybrid_do)
        c2d_2 = Conv1D(64, 3, activation = 'elu', kernel_initializer = init, padding = 'same',
                       name = 'met_c2d_2')(c2d_1)
        c2d_3 = Conv1D(64, 3, activation = 'elu', kernel_initializer = init, padding = 'same',
                       name = 'met_c2d_3')(c2d_2)
        c2d_4 = MaxPooling1D(pool_size = 3, strides = 1, padding = 'same', name = 'met_c2d_4')(c2d_3)
        c2d_5 = BatchNormalization(momentum = 0.99, name = 'met_c2d_5')(c2d_4)
        c2d_6 = Dropout(0.25, name = 'met_c2d_6')(c2d_5)

        ## conv1d block 4
        c4d_1 = Dense(32, kernel_initializer = init, activation = 'relu', name = 'met_c4d_1')(c2d_6)
        c4d_2 = Conv1D(32, 3, activation = 'relu', kernel_initializer = init, padding = 'same',
                       name = 'met_c4d_2')(c4d_1)
        c4d_3 = Conv1D(32, 3, activation = 'relu', kernel_initializer = init, padding = 'same',
                       name = 'met_c4d_3')(c4d_2)
        c4d_4 = MaxPooling1D(pool_size = 3, strides = 1, padding = 'same', name = 'met_c4d_4')(c4d_3)
        c4d_5 = BatchNormalization(momentum = 0.99, name = 'met_c4d_5')(c4d_4)
        c4d_6 = Dropout(0.25, name = 'c4d_6')(c4d_5)

        ## shortcut 1
        sc1_pre = Conv1D(32, 1, activation = 'relu', kernel_initializer = init, padding = 'same',
                         name = 'met_sc1_pre')(c2d_6)
        sc1 = Add(name = 'shortcut_1')([sc1_pre, c4d_6])

        ## conv1d block 5
        c5d_1 = Dense(16, kernel_initializer = init, activation = 'relu', name = 'met_c5d_1')(sc1)
        c5d_2 = Conv1D(16, 3, activation = 'relu', kernel_initializer = init, padding = 'same',
                       name = 'met_c5d_2')(c5d_1)
        c5d_3 = Conv1D(16, 3, activation = 'relu', kernel_initializer = init, padding = 'same',
                       name = 'met_c5d_3')(c5d_2)
        c5d_4 = MaxPooling1D(pool_size = 3, strides = 1, padding = 'same', name = 'met_c5d_4')(c5d_3)
        c5d_5 = BatchNormalization(momentum = 0.99, name = 'met_c5d_5')(c5d_4)
        c5d_6 = Dropout(0.25, name = 'met_c5d_6')(c5d_5)

        ## shortcut 2
        sc2_pre = Conv1D(16, 1, activation = 'relu', kernel_initializer = init, padding = 'same',
                         name = 'met_sc2_pre')(sc1)
        sc2 = Add(name = 'shortcut_2')([sc2_pre, c5d_6])
        c5d_7 = GlobalAveragePooling1D(name = 'met_c5d_7')(sc2)

        ## dense block and regression
        d1_1 = Dense(16, activation = 'elu', name = 'fusion_output')(c5d_7)
        # d1_2 = Dense(16, activation='linear')(d1_1)
        d1_3 = Dense(8, activation = 'elu', name = 'met_d1_3')(d1_1)
        inf_output = Dense(dp.shape_val[1], activation = 'softmax', name = 'met_inference')(d1_3)

        ## compile
        model = Model(inputs = input_meteor, outputs = inf_output)
        model._name = 'Meteor_Core'
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy',
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.Recall()])
        model.summary()

        self.model = model

    def load_tuned_model(self):
        eva_model = load_model('./meteor_' + str(self.acc_epoch) + '/model/meteor_core.h5')
        eva_model.summary()

        self.eva_model = eva_model


class BuildFusion:
    def __init__(self):
        self.dir_path = os.getcwd()

        print('now initiating the fusion on cores. ')
        self.current_epoch = int(1)
        self.acc_epoch = int(0)

        self._init_fusion_core()

    def _load_visual_core(self, target_model):
        _vis_core = load_model(target_model)
        _vis_core.summary()
        return _vis_core

    def _load_meteor_core(self, target_model):
        _met_core = load_model(target_model)
        _met_core.summary()
        return _met_core

    def _init_fusion_core(self):

        if seq_train_flag:
            global vis_core, met_core
            vis_core_dir = './visual_' + str(vis_core.acc_epoch) + '/model/visual_core.h5'
            met_core_dir = './meteor_' + str(met_core.acc_epoch) + '/model/meteor_core.h5'
            del vis_core, met_core
            gc.collect()
        else:
            vis_core_dir = os.path.join('./model/', 'visual_core.h5')
            met_core_dir = os.path.join('./model/', 'meteor_core.h5')

        _vis_core = self._load_visual_core(vis_core_dir)
        _met_core = self._load_meteor_core(met_core_dir)

        ## add an attention layer here
        vis_output = tf.expand_dims(_vis_core.get_layer('fc2').output, axis = 1)
        vis_weight = Conv1D(16, 1, activation = 'elu', padding = 'same')(vis_output)
        vis_att = Multiply()([vis_output, vis_weight])

        met_output = tf.expand_dims(_met_core.get_layer('fusion_output').output, axis = 1)
        met_weight = Conv1D(16, 1, activation = 'elu', padding = 'same')(met_output)
        met_att = Multiply()([met_output, met_weight])

        ## concatenate
        fuse_clip = tf.keras.layers.concatenate([vis_att, met_att], name = 'fuse_input', axis = 1)
        fuse_load = Dropout(0.2, name = 'fuse_dropout')(fuse_clip)

        fuse_unfold = GlobalAveragePooling1D(name = 'fuse_unfold')(fuse_load)

        fuse_fc_1 = Dense(8, activation = 'elu', name = 'fuse_fc_1')(fuse_unfold)
        fuse_fc_2 = Dense(8, activation = 'elu', name = 'fuse_fc_2')(fuse_fc_1)
        fuse_inf = Dense(dp.shape_val[1], activation = 'softmax', name = 'fuse_inference')(fuse_fc_2)

        fusion = Model(inputs = [_vis_core.input, _met_core.input],
                       outputs = fuse_inf)
        fusion._name = 'Fusion_Core'

        layer_cond = ('conv1')
        for layer in fusion.layers:
            if layer._name.startswith(layer_cond):
                layer.trainable = False

        fusion.summary()

        fusion.compile(loss = 'categorical_crossentropy',
                       optimizer = 'adam',
                       metrics = ['accuracy',
                                  tf.keras.metrics.Precision(),
                                  tf.keras.metrics.Recall()])

        self.fusion = fusion

    def load_tuned_model(self):
        eva_model = load_model('./fusion_' + str(self.acc_epoch) + '/model/fusion_core.h5')
        eva_model.summary()

        self.eva_model = eva_model


def create_folder(dest_dir, type_var = None):
    if type_var:
        dest_dir_parent = './' + str(type_var) + '_' + str(dest_dir)
    else:
        dest_dir_parent = './' + str(dest_dir)

    dest_dir_sub = str(dest_dir_parent) + '/' + 'model'

    if not os.path.exists(dest_dir_parent):
        os.makedirs(dest_dir_parent)

    if not os.path.exists(dest_dir_sub):
        os.makedirs(dest_dir_sub)


def train_visual_core_call(epoch):
    vis_core.current_epoch = int(epoch)
    print('from', str(vis_core.acc_epoch), 'epoch, start', str(vis_core.current_epoch), 'epoch. ')
    vis_core.acc_epoch += vis_core.current_epoch
    create_folder(vis_core.acc_epoch, 'visual')
    output_model_path = './visual_' + str(vis_core.acc_epoch) + '/model/' + 'visual_core.h5'

    checkpoint = ModelCheckpoint(output_model_path, monitor = 'val_loss',
                                 verbose = 1, save_best_only = True, mode = 'auto')

    tune = vis_core.visual_core.fit(x = dp.train_vis_part, y = dp.label_train,
                                    batch_size = dp.batch_size_vis,
                                    validation_split = 0.15, epochs = vis_core.current_epoch,
                                    shuffle = True, callbacks = [checkpoint], verbose = 1)
    evaluate_visual_core()
    pass


def train_meteor_core_call(epoch):
    met_core.current_epoch = int(epoch)
    print('from', str(met_core.acc_epoch), 'epoch, start', str(met_core.current_epoch), 'epoch. ')
    met_core.acc_epoch += met_core.current_epoch
    create_folder(met_core.acc_epoch, 'meteor')
    output_model_path = './meteor_' + str(met_core.acc_epoch) + '/model/meteor_core.h5'

    checkpoint = ModelCheckpoint(output_model_path, monitor = 'val_loss',
                                 verbose = 1, save_best_only = True, mode = 'auto')

    met_tune = met_core.model.fit(x = dp.train_met_part, y = dp.label_train,
                                  batch_size = dp.batch_size_met,
                                  validation_split = 0.15, epochs = met_core.current_epoch,
                                  shuffle = True, callbacks = [checkpoint], verbose = 1)
    evaluate_meteor_core()
    pass


def train_sub_models():
    ## the training of meteor model should be limited, 500 will be ideal
    train_meteor_core_call(20)

    del met_core.model, met_core.eva_model
    gc.collect()

    train_visual_core_call(200)

    del vis_core.visual_core, vis_core.eva_model
    gc.collect()
    pass


def train_fusion_core_call(epoch):
    fuse_core.current_epoch = int(epoch)
    print('FUSION CORE. from', str(fuse_core.acc_epoch), 'epoch, start', str(fuse_core.current_epoch), 'epoch. ')
    fuse_core.acc_epoch += fuse_core.current_epoch
    create_folder(fuse_core.acc_epoch)
    output_model_path = './fusion_' + str(fuse_core.acc_epoch) + '/model/' + 'fusion_core.h5'

    checkpoint = ModelCheckpoint(output_model_path, monitor = 'val_loss',
                                 verbose = 1, save_best_only = True, mode = 'auto')

    tune = fuse_core.fusion.fit({'visual_input': dp.train_vis_part,
                                 'meteor_input': dp.train_met_part},
                                dp.label_train, batch_size = dp.batch_size_vis,
                                validation_split = 0.15, epochs = fuse_core.current_epoch, shuffle = True,
                                callbacks = [checkpoint], verbose = 1)

    evaluate_fusion_core()


def train_fusion_core():
    train_fusion_core_call(90)
    pass


def evaluate_visual_core():
    print('evaluating the visual core. ')
    vis_core.load_tuned_model()
    output = vis_core.eva_model.predict(dp.valid_vis_part)

    ## add 1 is mainly because that the default argmax starts with 0, but in our case should be 1
    out_arg = np.argmax(output, axis = 1).reshape(-1, 1) + 1

    comb_arr = np.hstack((output, out_arg, dp.label_valid_raw.reshape(-1, 1)))
    comb_df = pd.DataFrame(comb_arr, columns = ['1', '2', '3', '4', '5', '6', 'est', 'label'])
    output_dir = './visual_' + str(vis_core.acc_epoch) + '/visual_core_infer_result.csv'
    comb_df.to_csv(output_dir, index = False)

    calc_performance_indicator(output_dir)


def evaluate_meteor_core():
    print('evaluating the meteor core. ')
    met_core.load_tuned_model()
    output = met_core.eva_model.predict(dp.valid_met_part)

    ## add 1 is mainly because that the default argmax starts with 0, but in our case should be 1
    out_arg = np.argmax(output, axis = 1).reshape(-1, 1) + 1

    comb_arr = np.hstack((output, out_arg, dp.label_valid_raw.reshape(-1, 1)))
    comb_df = pd.DataFrame(comb_arr, columns = ['1', '2', '3', '4', '5', '6', 'est', 'label'])
    output_dir = './meteor_' + str(met_core.acc_epoch) + '/meteor_core_infer_result.csv'
    comb_df.to_csv(output_dir, index = False)

    calc_performance_indicator(output_dir)


def evaluate_fusion_core():
    print('now starting the evaluation process. ')
    fuse_core.load_tuned_model()

    output = fuse_core.eva_model.predict({'visual_input': dp.valid_vis_part, 'meteor_input': dp.valid_met_part})

    ## add 1 is mainly because that the default argmax starts with 0, but in our case should be 1
    out_arg = np.argmax(output, axis = 1).reshape(-1, 1) + 1

    comb_arr = np.hstack((output, out_arg, dp.label_valid_raw.reshape(-1, 1)))
    comb_df = pd.DataFrame(comb_arr, columns = ['1', '2', '3', '4', '5', '6', 'est', 'label'])
    output_dir = './fusion_' + str(fuse_core.acc_epoch) + '/fusion_core_infer_result.csv'
    comb_df.to_csv(output_dir, index = False)

    calc_performance_indicator(output_dir)


def evaluate_fusion_core_on_quit():
    output = fuse_core.fusion.predict({'visual_input': dp.valid_vis_part, 'meteor_input': dp.valid_met_part})

    ## add 1 is mainly because that the default argmax starts with 0, but in our case should be 1
    out_arg = np.argmax(output, axis = 1).reshape(-1, 1) + 1

    current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    output_dir = './fusion_' + str(fuse_core.acc_epoch) + '/' + current_time + 'fusion_core_infer_result.csv'

    comb_arr = np.hstack((output, out_arg, dp.label_valid_raw.reshape(-1, 1)))
    comb_df = pd.DataFrame(comb_arr, columns = ['1', '2', '3', '4', '5', '6', 'est', 'label'])
    comb_df.to_csv(output_dir, index = False)

    calc_performance_indicator(output_dir)


def calc_performance_indicator(incoming_file):
    tar_file = str(incoming_file)
    sub_folder = tar_file.split('/')[-2]
    type_var = tar_file.split('_')[-4].split('/')[-1]

    file = pd.read_csv(tar_file)
    precision_result = mc.precision_score(file['label'], file['est'], average = None).reshape(-1, 1)
    recall_result = mc.recall_score(file['label'], file['est'], average = None).reshape(-1, 1)
    f1_result = mc.f1_score(file['label'], file['est'], average = None).reshape(-1, 1)

    agg_result = np.empty(shape = (f1_result.shape[0], 0))
    agg_result = np.hstack((agg_result, precision_result))
    agg_result = np.hstack((agg_result, recall_result))
    agg_result = np.hstack((agg_result, f1_result))
    type_index = np.array(['excel', 'good', 'light', 'moder', 'heavi', 'sever']).reshape(-1, 1)
    agg_result = np.hstack((type_index, agg_result))
    agg_table = pd.DataFrame(agg_result, columns = ['type', 'preci', 'recall', 'f1'])
    agg_table.to_csv('./' + str(sub_folder) + '/' + str(type_var) + '_performance.csv', index = False)


if __name__ == '__main__':

    ## please cite our work if this code benefits your research.
    ## > M. Yao, D. Tao, J. Wang, R. Gao and K. Sun,
    ## > "MARVAir: Meteorology Augmented Residual-Based Visual Approach for Crowdsourcing Air Quality Inference,"
    ## > in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-10, 2022, Art no. 2514310, doi: 10.1109/TIM.2022.3193197.

    server_flag = 'local'

    seq_train_flag = True

    dp = DataProcessor(server_flag)

    if seq_train_flag:
        vis_core = BuildVisualCore()

        met_core = BuildMeteorCore()

        train_sub_models()

    fuse_core = BuildFusion()

    train_fusion_core()

    del dp, fuse_core

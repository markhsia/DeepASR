# Python

import tensorflowjs as tfjs

import os

from keras.models import load_model
# 指定第?块GPU可用
# print(os.popen('nvidia-smi').read())
# which_GPU = input('which_GPU?')
# os.environ["CUDA_VISIBLE_DEVICES"] = which_GPU


load_weight_path = "saved_models/CNN1d_CTC_PinYin_Sample_lessDropout/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=70)(loss=7.7)(val_loss=10.5).keras_weight"


base_path = os.path.splitext(load_weight_path)[0]
predict_model_h5_path = base_path+"predict_model.h5"
tfjs_target_dir = base_path + "predict_model.tfjs"

def train():
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    model = load_model(predict_model_h5_path)
    model.summary()
    tfjs.converters.save_keras_model(model, tfjs_target_dir)

train()
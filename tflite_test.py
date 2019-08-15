from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample_lessDropout
from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type
import os

import tensorflow as tf
# 指定第?块GPU可用
# print(os.popen('nvidia-smi').read())
# which_GPU = input('which_GPU?')
# os.environ["CUDA_VISIBLE_DEVICES"] = which_GPU


load_weight_path = "saved_models/CNN1d_CTC_PinYin_Sample_lessDropout/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=70)(loss=7.7)(val_loss=10.5).keras_weight"


base_path = os.path.splitext(load_weight_path)[0]
predict_model_h5_path = base_path+"predict_model.h5"
converted_predict_model_tflite_path = base_path + "converted_predict_model.tflite"

if not os.path.exists(predict_model_h5_path):
    model_obj = CNN1d_CTC_PinYin_Sample_lessDropout('CNN1d_CTC_PinYin_Sample_lessDropout',feature=mel_fea, data_cache_dir=None,label_type=label_type,save_dir = model_save_dir)
    model_obj.load_weight(load_weight_path)
    print("创建predict_model的h5文件:",predict_model_h5_path)
    model_obj.predict_model.save(predict_model_h5_path)

os.environ["CUDA_VISIBLE_DEVICES"] = ''
converter = tf.lite.TFLiteConverter.from_keras_model_file(predict_model_h5_path,input_shapes = {"padded_datas" : [None,80,128]})
tflite_model = converter.convert()
with open(converted_predict_model_tflite_path, "wb") as f:
    f.write(tflite_model)



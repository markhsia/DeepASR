import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ''
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_models/CNN1d_CTC_PinYin_Sample_lessDropout/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=70)(loss=7.7)(val_loss=10.5)converted_predict_model.tflite")
interpreter.allocate_tensors()
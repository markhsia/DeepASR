from Scripts.Server.Server import Server
from Scripts.Data.AcousticData import PureAcousticData
from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample_Dropout2
from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type

model_obj = CNN1d_CTC_PinYin_Sample_Dropout2('CNN1d_CTC_PinYin_Sample_Dropout2(AiShell)',feature=mel_fea, data_cache_dir=None,label_type=label_type,save_dir = model_save_dir)
load_weight_path = "/home/A/Work/Speech/MyASR/saved_models/CNN1d_CTC_PinYin_Sample_Dropout2(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)_1/best_val_loss(epoch=239)(loss=10)(val_loss=9).keras_weight"
model_obj.load_weight(load_weight_path)
server = Server(model_obj)
server.run()
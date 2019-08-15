from Scripts.Server.Server import Server
from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample_lessDropout
from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type

model_obj = CNN1d_CTC_PinYin_Sample_lessDropout('CNN1d_CTC_PinYin_Sample_lessDropout',feature=mel_fea, data_cache_dir=None,label_type=label_type,save_dir = model_save_dir)
load_weight_path = "saved_models/CNN1d_CTC_PinYin_Sample_lessDropout/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=70)(loss=7.7)(val_loss=10.5).keras_weight"
model_obj.load_weight(load_weight_path)
server = Server(model_obj)
server.run()
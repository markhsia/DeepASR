
from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type, epochs, batch_size
from Scripts.Data.AiShellData import AiShellData
from Scripts.Data.AcousticData import PureAcousticData
from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
from Scripts.Models.CNN2d_CTC import CNN2d_CTC_PinYin
from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample_lessDropout

from Scripts.utils.others import make_AuDataObjs_list

from train import AcousticTrainer_OneData

base_path = '/data/speech/AiShell/data_aishell/wav/'
train_paths = [base_path + 'train']
dev_paths = [base_path + 'dev']
test_paths = [base_path + 'test']
if __name__ == '__main__':
    trainer = AcousticTrainer_OneData(
        DataOBJ = AiShellData,
        train_paths = train_paths,
        dev_paths = dev_paths,
        test_paths = test_paths,
        feature = mel_fea,# !!!
        data_cache_dir = cache_dir,
        label_type = 'pinyin',
        ModelOBJ = CNN1d_CTC_PinYin_Sample_lessDropout,# !!!
        Model_name = 'CNN1d_CTC_PinYin_Sample_lessDropout(AiShell)',# !!!
        epochs = 700,
        batch_size = batch_size,
        patience = 200,
        model_save_dir = model_save_dir,
        debug = False,
        debug_data_num = 100,
        debug_model_save_dir = 'debug/saved_models',
        debug_epochs = 5,
        )
    trainer.train_and_test(load_weight_path=None)
    # trainer.load_and_test(load_weight_path='saved_models/CNN1d_CTC_PinYin_Sample_Dropout(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=189)(val_loss=15).keras_weight')

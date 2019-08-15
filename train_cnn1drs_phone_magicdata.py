from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type,batch_size
from Scripts.Data.MagicData import MagicData
from Scripts.Models.CNN1d_CTC import CNN1d_CTC_Phone_RealSample

from train import AcousticTrainer_OneData

base_path = '/data/speech/MAGICDATA/'
train_paths = [base_path + 'train']
dev_paths = [base_path + 'dev']
test_paths = [base_path + 'test']

if __name__ == '__main__':
    trainer = AcousticTrainer_OneData(
        data_name = 'MagicData',
        DataOBJ = MagicData,
        train_paths = train_paths,
        dev_paths = dev_paths,
        test_paths = test_paths,
        feature = mel_fea,# !!!
        data_cache_dir = cache_dir,
        label_type = 'phone',
        ModelOBJ = CNN1d_CTC_Phone_RealSample,# !!!
        Model_name = 'CNN1d_CTC_Phone_RealSample',# !!!
        epochs = 500,
        batch_size = batch_size,
        patience = 20,
        model_save_dir = model_save_dir,
        debug = False,
        debug_data_num = 100,
        debug_model_save_dir = 'debug/saved_models',
        debug_epochs = 5,
        )
    trainer.train_and_test(load_weight_path=None)
    # load_weight_path = 'saved_models/CNN1d_CTC_PinYin_Sample/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=35)(loss=5.7)(val_loss=11.2).keras_weight'
    # trainer.load_and_test(load_weight_path=load_weight_path)
    # trainer.manully_test(['/home/A/Work/Speech/MyDeepASR_old/datas/mytest_t/'],load_weight_path = load_weight_path)

from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type,batch_size
from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample

from train import AcousticTrainer


from get_all_datas import train_DataObjs,dev_DataObjs,test_DataObjs


if __name__ == '__main__':
    trainer = AcousticTrainer(
        train_DataObjs,
        dev_DataObjs,
        test_DataObjs,
        feature = mel_fea,# !!!
        data_cache_dir = cache_dir,
        label_type = 'pinyin',
        ModelOBJ = CNN1d_CTC_PinYin_Sample,# !!!
        Model_name = 'CNN1d_CTC_PinYin_Sample(LargeData)',# !!!
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
    # load_weight_path = "/home/A/Work/Speech/MyASR/saved_models/CNN1d_CTC_PinYin_Sample(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/final(epoch=119)(loss=2.4)(val_loss=18.0).keras_weight"
    # trainer.load_and_test(load_weight_path=load_weight_path)
    # trainer.manully_test(['/home/A/Work/Speech/MyDeepASR_old/datas/mytest_t/'],load_weight_path = load_weight_path)

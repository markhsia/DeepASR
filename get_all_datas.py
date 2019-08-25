# 导入为不同数据集写的数据类
from Scripts.Data.AcousticData import PureAcousticData
from Scripts.Data.AiShellData import AiShellData
from Scripts.Data.AiDataTang import AiDataTang
from Scripts.Data.MagicData import MagicData
from Scripts.Data.PrimewordsData import PrimewordsData
from Scripts.Data.Thchs30Data import Thchs30Data
from Scripts.Data.ST_CMDSData import ST_CMDSData

# 一个辅助函数，方便地收集一个文件夹下的所有音频文件，无需事先把文件名列好
from Scripts.utils.others import make_AuDataObjs_list

#列出各个数据集的根目录来
Aishell_base_path = '/data/speech/AiShell/data_aishell/wav/'

Aidatatang_base_path = '/data/speech/Aidatatang/aidatatang_200zh/corpus/'

Thchs30_base_path = '/data/speech/thchs30/openslr_format/data_thchs30/'

MagicData_base_path = '/data/speech/MAGICDATA/'

ST_CMDS_path = "/data/speech/ST-CMDS/ST-CMDS-20170001_1-OS/"

Primewords_path = '/data/speech/Primewords/primewords_md_2018_set1/audio_files/'

# 初始化保存音频数据对象的列表
train_DataObjs = []
dev_DataObjs = []
test_DataObjs = []

# AiShell，AiDataTang，Thchs30数据集都自带分好了train，dev，test文件夹
for (DataOBJ,base_path) in (
        (AiShellData,Aishell_base_path),
        (AiDataTang,Aidatatang_base_path),
        (Thchs30Data,Thchs30_base_path),
        (MagicData,MagicData_base_path),
    ):
    print("列出%s样本"%(DataOBJ.__name__))
    trains = make_AuDataObjs_list(DataOBJ, paths=[base_path+'train/'])
    devs = make_AuDataObjs_list(DataOBJ, paths=[base_path+'dev/'])
    tests = make_AuDataObjs_list(DataOBJ, paths=[base_path+'test/'])
    print("训练集%d个\t验证集%d个\t测试集%d个\n"%(len(trains),len(devs),len(tests)))
    train_DataObjs += trains
    dev_DataObjs += devs
    test_DataObjs += tests

# ST_CMDS，Primewords数据集没有事先分好train，dev，test，因此手动分割
split_k = (0.9,0.3)
for DataOBJ,path in (
        (ST_CMDSData,ST_CMDS_path),
        (PrimewordsData,Primewords_path),
    ):
    print("列出%s样本"%(DataOBJ.__name__))
    DataObjs = make_AuDataObjs_list(DataOBJ, paths=[path])
    train_N = int(len(DataObjs)*split_k[0])
    dev_N = int((len(DataObjs) - train_N) * split_k[1])
    print("训练集%d个\t验证集%d个\t测试集%d个\n"%(train_N,dev_N,len(DataObjs) - train_N - dev_N ))
    train_DataObjs += DataObjs[:train_N]
    dev_DataObjs += DataObjs[train_N:train_N+dev_N]
    test_DataObjs += DataObjs[train_N+dev_N:]
    

if __name__ == '__main__':
    from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type, batch_size
    from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
    dataparser = AcousticDataParser(feature=mel_fea, cache_dir=cache_dir)
    labelparser = AcousticLabelParser(label_type=label_type)

    from Scripts.utils.others import test_dataObjs

    test_dataObjs(train_DataObjs+dev_DataObjs+test_DataObjs,dataparser,labelparser,'Magic_ERROR_DATA_INFOS.jsons')
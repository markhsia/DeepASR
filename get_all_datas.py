# 导入为不同数据集写的数据类
from Scripts.Data.AcousticData import PureAcousticData
from Scripts.Data.AiShellData import AiShellData
from Scripts.Data.AiDataTang import AiDataTang
from Scripts.Data.PrimewordsData import PrimewordsData
from Scripts.Data.Thchs30Data import Thchs30Data
from Scripts.Data.ST_CMDSData import ST_CMDSData

# 一个辅助函数，方便地收集一个文件夹下的所有音频文件，无需事先把文件名列好
from Scripts.utils.others import make_AuDataObjs_list

#列出各个数据集的根目录来
Aishell_base_path = '/data/speech/AiShell/data_aishell/wav/'

Aidatatang_base_path = '/data/speech/Aidatatang/aidatatang_200zh/corpus/'

Thchs30_base_path = '/data/speech/thchs30/openslr_format/data_thchs30/'

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
    # 直接执行此文件将遍历测试一遍所有的数据，
    # 有问题的单个数据id会自动被ERROR_DATA_INFO类持久化记录下来到一个文件中去，以后make_AuDataObjs_list载入时可选择略过
    class ShorterThanLabelError(Exception):
        '''当音频长度过短时,raise这个类
        '''
        pass
    from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
    import json,traceback
    from config import model_save_dir, cache_dir, feature, stft_fea, mel_fea, label_type,batch_size
    from tqdm import tqdm
    from Scripts.utils.others import ERROR_DATA_INFO
    error_info = ERROR_DATA_INFO()
    dataparser = AcousticDataParser(feature=mel_fea, cache_dir=cache_dir)
    labelparser = AcousticLabelParser(label_type=label_type)
    from tqdm import tqdm
    for data_obj in tqdm(train_DataObjs + dev_DataObjs + test_DataObjs):
        data = None
        label = None
        try:
            data = dataparser(data_obj)
            label = labelparser(data_obj)
            if data.shape[0]//8 < len(label):
                raise ShorterThanLabelError("音频长度%d//8 小于 标签长度%d"%(data.shape[0],len(label)))
        except Exception as e:
            print("\n=================\n出错data为:",data_obj.id)
            error_info.save_error_info(data_obj.id)
            print("出错数据shape:",data.shape)
            print("出错数据parsed_label为:",label)
            try:
                print("出错数据文字为:",data_obj._get_one_text(data_obj.filepath))
                print("出错数据label为:",data_obj.get_label())
            except:
                pass
            print("\t****错误信息****")
            print(traceback.format_exc().replace('\n','\n\t'))
            print("\n=================\n")
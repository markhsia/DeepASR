from Scripts.Data.AcousticData import PureAcousticData
from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
from Scripts.utils.others import make_AuDataObjs_list
from Scripts.utils.tools import format_dict2file
from Scripts.Models.AcousticModel import AcousticModel
import json,os
class AcousticTrainer:
    def __init__(self,
        data_name:str,
        train_DataObjs,
        dev_DataObjs,
        test_DataObjs,
        feature,
        data_cache_dir,
        label_type,
        ModelOBJ:AcousticModel,
        Model_name,
        epochs,
        batch_size,
        patience,
        model_save_dir = 'saved_models',
        debug = False,
        debug_data_num = 100,
        debug_model_save_dir = 'debug/saved_models',
        debug_epochs = 5,
        ):
        # 准备数据
        self.data_name = data_name
        self.train_DataObjs = train_DataObjs
        self.dev_DataObjs = dev_DataObjs
        self.test_DataObjs = test_DataObjs

        # 准备模型
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        if debug:
            self.train_DataObjs = self.train_DataObjs[:debug_data_num]
            self.dev_DataObjs = self.dev_DataObjs[:debug_data_num]
            self.test_DataObjs = self.test_DataObjs[:debug_data_num]
            self.epochs = debug_epochs
            model_save_dir = debug_model_save_dir
            print()
            print("="*10)
            print("***DEBUG模式***\n训练数据量:%d\n验证数据量:%d\n测试数据量:%d" %
                (len(self.train_DataObjs), len(self.dev_DataObjs), len(self.test_DataObjs)))
            print("训练轮数epochs:", self.epochs)
            print("模型保存文件夹:", model_save_dir)
            print("="*10)
            input("任意键继续")
        
        self.model_obj = ModelOBJ(Model_name, feature=feature, data_cache_dir=data_cache_dir,label_type=label_type, save_dir = model_save_dir)

    def train_and_test(self,load_weight_path=None):
        if load_weight_path:
            self.model_obj.load_weight(load_weight_path)
            print("读取模型权重:",load_weight_path)
        print("训练模型")
        self.model_obj.fit(self.data_name,self.train_DataObjs, self.dev_DataObjs, batch_size=self.batch_size, epochs=self.epochs,patience = self.patience)
        
        print("测试模型")
        self.model_obj.test(self.test_DataObjs, batch_size=16, pre_n=5)

        res = self.model_obj.predict(PureAcousticData(
            "/home/A/Work/Speech/MyDeepASR_old/datas/mytest_t/A11_0.wav"))
        print('\n自定义识别:', ' '.join(res))


    def load_and_test(self, load_weight_path = None,wer_test_n = 100,greedy=True, beam_width=100):
        if load_weight_path:
            self.model_obj.load_weight(load_weight_path)
            print("读取模型权重:",load_weight_path)
        print("=====测试模型=====")
        print("\n测试2000训练集")
        self.model_obj.test(self.train_DataObjs[:2000], batch_size=16, pre_n=1)
        print("\n测试验证集")
        self.model_obj.test(self.dev_DataObjs, batch_size=16, pre_n=1)
        print("\n测试测试集")
        self.model_obj.test(self.test_DataObjs, batch_size=16, pre_n=1)
        print("\n测试训练集%d个WER"%wer_test_n)
        self.model_obj.test_wer(self.train_DataObjs[:wer_test_n],greedy, beam_width)
        print("\n测试验证集%d个WER"%wer_test_n)
        self.model_obj.test_wer(self.dev_DataObjs[:wer_test_n],greedy, beam_width)
        print("\n测试测试集%d个WER"%wer_test_n)
        self.model_obj.test_wer(self.test_DataObjs[:wer_test_n],greedy, beam_width)

        res = self.model_obj.predict(PureAcousticData(
            "/home/A/Work/Speech/MyDeepASR_old/datas/mytest_t/A11_0.wav"))
        print('\n自定义识别:', ' '.join(res))
    
    def manully_test(self,data_paths, load_weight_path = None ):
        '''data_paths:包含自定义音频的文件路径或目录路径的列表,该路径列表将由make_AuDataObjs_list处理'''
        if load_weight_path:
            self.model_obj.load_weight(load_weight_path)
            print("读取模型权重:",load_weight_path)        
        DataObjs = make_AuDataObjs_list(PureAcousticData, paths=data_paths)
        for data_obj in DataObjs:
            res = self.model_obj.predict(data_obj)
            print('\n自定义识别文件:%s\n识别结果:%s'%(data_obj.filepath, ' '.join(res)))       


class AcousticTrainer_OneData(AcousticTrainer):
    def __init__(self,
        data_name:str,
        DataOBJ,
        train_paths,
        dev_paths,
        test_paths,
        feature,
        data_cache_dir,
        label_type,
        ModelOBJ:AcousticModel,
        Model_name,
        epochs,
        batch_size,
        patience,
        model_save_dir = 'saved_models',
        debug = False,
        debug_data_num = 100,
        debug_model_save_dir = 'debug/saved_models',
        debug_epochs = 5,
        ):
        # 准备数据
        train_DataObjs = make_AuDataObjs_list(DataOBJ, paths=train_paths)
        dev_DataObjs = make_AuDataObjs_list(DataOBJ, paths=dev_paths)
        test_DataObjs = make_AuDataObjs_list(DataOBJ, paths=test_paths)

        super().__init__(
            data_name,
            train_DataObjs,
            dev_DataObjs,
            test_DataObjs,
            feature,
            data_cache_dir,
            label_type,
            ModelOBJ,
            Model_name,
            epochs,
            batch_size,
            patience,
            model_save_dir,
            debug,
            debug_data_num,
            debug_model_save_dir,
            debug_epochs 
            )

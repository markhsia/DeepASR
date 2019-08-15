from .BaseModel import KerasCTCBaseModel
from .DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
from abc import abstractmethod
from ..utils.wer import wer
from tqdm import tqdm
class AcousticModel(KerasCTCBaseModel):

    def __init__(self, name, feature, data_cache_dir,label_type, save_dir='saved_model'):
        dataparser = AcousticDataParser(feature=feature, cache_dir=data_cache_dir)
        labelparser = AcousticLabelParser(label_type=label_type)
        super().__init__(name, dataparser,labelparser,save_dir)
       
    @abstractmethod
    def _create_fit_infos_dict(self,data_name,train_DataObjIter,dev_DataObjIter,batch_size,epochs):
        fit_infos_dict = {
            'gpu_n':self.gpu_num,
            'data_name': data_name,
            'train_data_num':len(train_DataObjIter),
            'dev_data_num':len(dev_DataObjIter),
            'feature':self.DataParser.feature,
            'label_type':self.LabelParser.label_type,
            'ModelOBJ_name':self.__class__.__name__,
            'model_name':self.name,
            'max_epochs':epochs,
            'batch_size':batch_size,
            'load_weight_path':self.load_weight_path,
            'main_save_dir':self.main_save_dir,
        }
        abstract_infos_dict = {
            'gpu_n':self.gpu_num,
            'feature_name' : self.DataParser.feature['name'],
            'label_type' :self.LabelParser.label_type        
        }
        return fit_infos_dict,abstract_infos_dict
        
    @abstractmethod
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''

        raise NotImplementedError
    
    def test(self, test_DataObjIter, batch_size, pre_n = 1):
        self.test_loss(test_DataObjIter, batch_size)
        self.test_detail(test_DataObjIter[:pre_n])
    
    def test_loss(self, test_DataObjIter, batch_size):
        test_batch_gen = self._batch_gen4DataObjIter(test_DataObjIter,batch_size)
        print('\n验证（共%d个音频)' % (len(test_DataObjIter)))
        print('验证loss:', self.model.evaluate_generator(generator=test_batch_gen))        
    
    def test_detail(self,test_DataObjIter,greedy=True, beam_width=100):
        wer_sum = 0
        test_n = len(test_DataObjIter)
        for data_obj in test_DataObjIter:
            true_label = data_obj.get_label(self.LabelParser.label_type)
            pred_label = self.predict(data_obj,greedy, beam_width)
            print('\n==========')
            print('识别音频文件：%s' % data_obj.filepath)
            print('正确结果:', ' '.join(true_label))
            print('识别结果：', ' '.join(pred_label))
            w = wer(true_label,pred_label)
            print('字错误率(WER):',w)
            wer_sum += w
            print("----------\n")
        if test_n>1:
            print("平均字错误率:",wer_sum/test_n)
    
    def test_wer(self,test_DataObjIter,greedy=True, beam_width=100):
        wer_sum = 0
        for data_obj in tqdm(test_DataObjIter):
            true_label = data_obj.get_label(self.LabelParser.label_type)
            pred_label = self.predict(data_obj,greedy, beam_width)        
            w = wer(true_label,pred_label)
            wer_sum += w
        print("平均字错误率:",wer_sum/len(test_DataObjIter))
from .BaseData import BaseData
import librosa
from abc import abstractmethod

class PureAcousticData(BaseData):
    def __init__(self, filepath):
        super().__init__(dataid=filepath)
        self.filepath = filepath

    def get_data(self):
        ori_y, ori_sr = librosa.load(self.filepath, sr=None)
        return ori_y, ori_sr
    
    def get_label(self,label_type):
        return []


class AcousticData(PureAcousticData):

    def __init__(self, filepath):
        super().__init__(filepath=filepath)
        
        self.get_label_funcs ={
            'text':self._get_one_text,
            'pinyin':self._get_one_pinyin_list,
            'phone':self._get_one_phone_list
        }

    def get_label(self,label_type):
        return self.get_label_funcs[label_type](self.filepath)

    @abstractmethod
    def _get_one_text(self, audio_fp):
        '''
        return 一个中文字符串,不能有任何标点符号或空白符
        '''
        pass

    def _get_one_pinyin_list(self, audio_fp):
        from ..Data.LabelType.pinyinbase import text2pinyin_list
        text = self._get_one_text(audio_fp)
        pinyin_list = text2pinyin_list(text)
        return pinyin_list

    def _get_one_phone_list(self, audio_fp):
        from ..Data.LabelType.phonebase import pinyin_list2phone_list
        pinyin_list = self._get_one_pinyin_list(audio_fp)
        phone_list = pinyin_list2phone_list(pinyin_list)
        return phone_list

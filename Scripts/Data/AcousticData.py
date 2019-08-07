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
    
    def get_label(self):
        return None


class AcousticData(PureAcousticData):

    def __init__(self, filepath, label_type='pinyin'):
        super().__init__(filepath=filepath)

        # label
        self.label_type = label_type
        if self.label_type == 'pinyin':
            self.get_label_func = self._get_one_pinyin_list
        elif self.label_type == 'phone':
            self.get_label_func = self._get_one_phone_list
        else:
            raise Exception("LabelType:%s 不存在" % self.label_type)

    def get_label(self):
        return self.get_label_func(self.filepath)

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

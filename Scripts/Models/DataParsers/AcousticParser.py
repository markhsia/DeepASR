
from .BaseParser import BaseDataParser, BaseLabelParser
from ...Data.AcousticData import AcousticData as AuDataObj,PureAcousticData
import os
import numpy as np
from ...utils.tools import _md5
from .Features.stft import _parse_one_audio_stft
from .Features.mfcc import _parse_one_audio_melspec
# from ...utils.tools import _md5,walk_subfiles

# AUDIO_TYPEs = ('.wav','.mp3','.aac')
# class AuDataObjIterator(BaseDataObjIterator):
#     def add_DataObjs(self, DataObj_class: AuDataObj, paths: list([str])):
#         super().add_DataObjs(DataObj_class, paths)

#     @staticmethod
#     def _make_DataObj_gen(DataObj_class: AuDataObj, paths: list([str])):
#         for fp in walk_subfiles(paths):
#             if os.path.splitext(fp)[1] in AUDIO_TYPEs:
#                 yield DataObj_class(filepath=fp)

class TooShortError(Exception):
    pass

class AcousticDataParser(BaseDataParser):

    def __init__(self, feature:dict, cache_dir:str=None, open_cache=True, re_cache=False):
        '''
        feature : {'name':feature_name, 'kwargs':feature_func_kwargs, 'feature_length':int}

        重要属性:self.AUDIO_FEATURE_LENGHT
        '''
        # cache
        self.main_cache_dir = cache_dir
        self.open_cache = open_cache if self.main_cache_dir else False
        self.re_cache = re_cache if self.open_cache else False
        if self.open_cache:
            sub_dir = feature['name']
            for k,v in feature['kwargs'].items():
                sub_dir += '(%s=%s)'%(k,v)
            self.cur_cache_dir = os.path.join(self.main_cache_dir,sub_dir)
            print("开启缓存!\n\t缓存主文件夹:%s\n\t当前缓存文件夹:%s"%(self.main_cache_dir,self.cur_cache_dir))
            if not os.path.exists(self.cur_cache_dir):
                print("创建文件夹:",self.cur_cache_dir)
                os.makedirs(self.cur_cache_dir)
        # data
        Feature_funcs = {
            'stft': _parse_one_audio_stft,
            'mel': _parse_one_audio_melspec
        }
        self.feature = feature
        if feature['name'] in Feature_funcs:
            self.feature_func = Feature_funcs[feature['name']]
            example_audio_Obj = PureAcousticData(os.path.join(os.path.dirname(__file__),'A11_0.wav'))
            print("使用样例数据:",example_audio_Obj.id)
            self.AUDIO_FEATURE_LENGTH = self.parse_one_data(
                example_audio_Obj).shape[1]
            print(
                "使用特征:", feature['name'],
                "特征参数:", feature['kwargs'],
                "特征长度:", self.AUDIO_FEATURE_LENGTH)
        else:
            raise Exception(
                "AcousticDataParser.Feature_funcs不存在%s特征" % feature['name'])
        
        input("请确认以上信息,按任意键继续...")


    def parse_one_data(self, au_data_obj: AuDataObj):
        if self.open_cache:
            cache_fp = self._get_cachefp(au_data_obj.filepath)
            if (not self.re_cache) and os.path.exists(cache_fp):
                try:
                    parsed_data = self._read_cache(cache_fp)
                except Exception as e:
                    print("读取缓存错误:",repr(e))
                    print("重新生成该缓存:",cache_fp)
                    os.remove(cache_fp)
                    parsed_data = self.feature_func(*au_data_obj.get_data(), **self.feature['kwargs'])
                    self._save_cache(parsed_data, cache_fp)
            else:
                parsed_data = self.feature_func(*au_data_obj.get_data(), **self.feature['kwargs'])
                self._save_cache(parsed_data, cache_fp)
        else:
            parsed_data = self.feature_func(*au_data_obj.get_data(), **self.feature['kwargs'])
        self._check_parsed_data(parsed_data)
        return parsed_data

    def _get_cachefp(self, data_fp):
        dirname = os.path.dirname(os.path.abspath(data_fp))
        dirname = dirname.strip('/\\').replace(":","")
        # sub_dir = dirname.replace("#","##").replace("/","#").replace("\\","#").replace(":","#")

        this_cache_dir = os.path.join(self.cur_cache_dir,dirname)
        if not os.path.exists(this_cache_dir):
            os.makedirs(this_cache_dir)
            # with open(os.path.normpath(os.path.join(cur_cache_dir,'/../','_info.txt')),'a') as f:
            #     f.write("dirname:%s"%dirname)
        cache_fp = os.path.join(
            this_cache_dir, os.path.basename(data_fp)+'.npy')
        return cache_fp

    @staticmethod
    def _read_cache(cache_fp):
        return np.load(cache_fp)

    @staticmethod
    def _save_cache(parsed_data, cache_fp):
        return np.save(cache_fp, parsed_data)
    
    @staticmethod
    def _check_parsed_data(parsed_data):
        # if parsed_data.shape[0] < 80:
        #     raise TooShortError
        pass


class AcousticLabelParser(BaseLabelParser):
    def __init__(self, label_type='pinyin'):
        ''' label_type : str

            重要属性:self.LABEL_NUM
        '''
        # label
        self.label_type = label_type
        if self.label_type == 'pinyin':
            from ...Data.LabelType.pinyinbase import pinyin2num, num2pinyin, pinyin_NUM
            self.encode_label_func = pinyin2num
            self.decode_label_func = num2pinyin
            self.LABEL_NUM = pinyin_NUM
        elif self.label_type == 'phone':
            from ...Data.LabelType.phonebase import phone2num, num2phone, phone_NUM
            self.encode_label_func = phone2num
            self.decode_label_func = num2phone
            self.LABEL_NUM = phone_NUM
        else:
            raise Exception("LabelType:%s 不存在" % self.label_type)

    def parse_one_label(self, au_data_obj: AuDataObj):
        try:
            res = self.encode_label(au_data_obj.get_label())
        except Exception as e:
            try:
                print("出错数据为:%s"%(au_data_obj.id))
                print("出错数据文字为:",au_data_obj._get_one_text(au_data_obj.filepath))
                print("出错数据label为:",au_data_obj.get_label())
            except:
                pass
            raise e
        return res

    def encode_label(self, label):
        if label is None:
            return [0]
        return self.encode_label_func(label)

    def decode_label(self, encoded_label):
        return self.decode_label_func(encoded_label)

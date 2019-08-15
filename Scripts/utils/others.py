from ..Data.AcousticData import AcousticData as AuDataObj
from .tools import _md5,walk_subfiles
import os
import json
import sys
import traceback
from tqdm import tqdm
class ERROR_DATA_INFO:
    def __init__(self,error_info_save_fp:str = 'ERROR_DATA_INFOs.jsons'):

        if not os.path.exists(error_info_save_fp):
            with open(error_info_save_fp,'w',encoding = 'utf-8'):
                pass
        self.error_info_save_fp = error_info_save_fp
        self.error_infos = self.load_error_infos()


    def save_error_info(self,data_obj):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        cur_error_infos = (data_obj.id,data_obj.__class__.__name__,exc_type.__name__,str(exc_value),traceback.format_exc())
        if data_obj.id not in self.error_infos:
            self.error_infos[data_obj.id] = cur_error_infos[1:]
            print("\t保存错误信息到:",self.error_info_save_fp)
            with open(self.error_info_save_fp,'a',encoding = 'utf-8') as f:
                f.write(json.dumps(cur_error_infos)+'\n')
        else:
            print("错误的data_id已存在")
            assert self.error_infos[data_obj.id][1] == exc_type.__name__,"%s %s\n != %s\n"%(data_obj.id,str(self.error_infos[data_obj.id]),str(cur_error_infos[1:]))

    def load_error_infos(self):
        error_datainfos = dict()
        print("载入%s文件中的错误data信息..."%self.error_info_save_fp)
        with open(self.error_info_save_fp,'r',encoding='utf-8') as f:
            for line in f:
                data_id,data_type,errortype,errorstr,desc = json.loads(line)
                error_datainfos[data_id] = (data_type,errortype,errorstr,desc)
        print("共载入%d条"%len(error_datainfos))
        return error_datainfos

error_data_info_obj = ERROR_DATA_INFO()
AUDIO_TYPEs = ('.wav','.mp3','.aac')
def make_AuDataObjs_gen(DataObj_class: AuDataObj, paths: list([str]), ignore_error_history = False):
    if ignore_error_history:
        error_datainfos = None 
    else:
        print("跳过错误数据信息表里的数据...")
        error_datainfos = error_data_info_obj.error_infos
    for fp in walk_subfiles(paths):
        if os.path.splitext(fp)[1] in AUDIO_TYPEs:
            if ignore_error_history or (fp not in error_datainfos):
                yield DataObj_class(filepath=fp)

def make_AuDataObjs_list(DataObj_class: AuDataObj, paths: list([str])):
    return [data_obj for data_obj in make_AuDataObjs_gen(DataObj_class, paths)]


def test_dataObjs(DataObjs,dataparser,labelparser,error_info_save_fp):
    # 直接执行此文件将遍历测试一遍所有的数据，
    # 有问题的单个数据id会自动被ERROR_DATA_INFO类持久化记录下来到一个文件中去，以后make_AuDataObjs_list载入时可选择略过
    class ShorterThanLabelError(Exception):
        '''当音频长度过短时,raise这个类
        '''
        pass
    error_info = ERROR_DATA_INFO(error_info_save_fp)
    for data_obj in tqdm(DataObjs):
        data = None
        label = None
        try:
            data = dataparser(data_obj)
            label = labelparser(data_obj)
            if data.shape[0]//8 < len(label):
                raise ShorterThanLabelError("音频长度%d//8 小于 标签长度%d"%(data.shape[0],len(label)))
        except Exception:
            print("\n=================\n出错data为:",data_obj.id)
            error_info.save_error_info(data_obj.id)
            try:
                print("出错数据shape:",data.shape if data else None)
                print("出错数据parsed_label为:",label)
                print("出错数据文字为:",data_obj._get_one_text(data_obj.filepath))
                print("出错数据label为:",data_obj.get_label(labelparser.label_type))
            except:
                pass
            print("\t****错误信息****")
            print(traceback.format_exc().replace('\n','\n\t'))
            print("\n=================\n")


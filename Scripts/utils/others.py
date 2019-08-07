from ..Data.AcousticData import AcousticData as AuDataObj
from .tools import _md5,walk_subfiles
import os
import json,sys

class ERROR_DATA_INFO:
    def __init__(self,error_info_save_fp:str = 'ERROR_DATA_INFOs.csv'):

        if not os.path.exists(error_info_save_fp):
            with open(error_info_save_fp,'w',encoding = 'utf-8') as f:
                pass
        self.error_info_save_fp = error_info_save_fp
        self.error_infos = self.load_error_infos()

    def save_error_info(self,data_id):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        cur_error_infos = (data_id,exc_type.__name__,str(exc_value),traceback.format_exc())
        if data_id not in self.error_infos:
            self.error_infos[data_id] = cur_error_infos[1:]
            print("\t保存错误信息到:",self.error_info_save_fp)
            with open(self.error_info_save_fp,'a',encoding = 'utf-8') as f:
                f.write(json.dumps(cur_error_infos)+'\n')
        else:
            print("错误的data_id已存在")
            assert self.error_infos[data_id][0] == exc_type.__name__,"%s %s\n != %s\n"%(data_id,str(self.error_infos[data_id]),str(cur_error_infos[1:]))

    def load_error_infos(self):
        error_datainfos = dict()
        with open(self.error_info_save_fp,'r',encoding='utf-8') as f:
            for line in f:
                data_id,errortype,errorstr,desc = json.loads(line)
                error_datainfos[data_id] = (errortype,errorstr,desc)
        return error_datainfos

AUDIO_TYPEs = ('.wav','.mp3','.aac')
def make_AuDataObjs_gen(DataObj_class: AuDataObj, paths: list([str]), ignore_error_history = True):
    error_datainfos = None if ignore_error_history else ERROR_DATA_INFO().error_infos
    for fp in walk_subfiles(paths):
        if os.path.splitext(fp)[1] in AUDIO_TYPEs:
            if ignore_error_history or (fp not in error_datainfos):
                yield DataObj_class(filepath=fp)

def make_AuDataObjs_list(DataObj_class: AuDataObj, paths: list([str])):
    return [data_obj for data_obj in make_AuDataObjs_gen(DataObj_class, paths)]

import traceback
from tqdm import tqdm
def flite_vailed_dataObjs(DataObjs,dataparser,labelparser):
    for data_obj in tqdm(DataObjs):
        try:
            data = dataparser(data_obj)
            label = labelparser(data_obj)
        except Exception as e:
            data = None
            label = None
            traceback.print_exc()
            err_fp = data_obj.filepath+'_'+repr(e)
            print(data_obj.id,"重命名为:",err_fp)
            os.rename(data_obj.filepath,err_fp)



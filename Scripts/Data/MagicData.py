import os
import json
from .AcousticData import AcousticData
from typing import List
magic_train_trans_fp = "/data/speech/MAGICDATA/train/TRANS.txt"
magic_dev_trans_fp = "/data/speech/MAGICDATA/dev/TRANS.txt"
magic_test_trans_fp = "/data/speech/MAGICDATA/test/TRANS.txt"

def _read_mutitrans(trans_fps:List[str]):
    trans_dict = dict()
    for trans_fp in trans_fps:
        trans_dict[trans_fp] = dict()
        with open(trans_fp,'r',encoding='utf8') as f:
            trans_list = f.read().split('\n')
            if trans_list[0] == 'UtteranceID\tSpeakerID\tTranscription\n':
                del trans_list[0]
            if trans_list[-1] == '':
                del trans_list[-1]
        for tran_line in trans_list:
            file_name,speakerID,text = tran_line.split('\t')
            trans_dict[trans_fp][file_name] = text

    return trans_dict

class MagicData(AcousticData):
    trans_dict = _read_mutitrans([magic_train_trans_fp,magic_dev_trans_fp,magic_test_trans_fp])

    def _get_one_text(self, audio_fp):
        dirname,basename = os.path.split(audio_fp)
        trans_fp = os.path.normpath(os.path.join(dirname,'..','TRANS.txt'))
        return self.trans_dict[trans_fp][basename]


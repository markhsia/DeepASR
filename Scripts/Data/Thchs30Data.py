from .AcousticData import AcousticData
from ..utils.tools import joinpath, abspath
import os


class Thchs30Data(AcousticData):
    def _get_one_text(self, audio_fp):
        '''跳过中文部分'''
        return

    def _get_one_pinyin_list(self, audio_fp):
        audio_fp_dir, audio_name = os.path.split(abspath(audio_fp))
        label_filepath = joinpath(audio_fp_dir, '../data', audio_name+'.trn')

        with open(label_filepath, 'r', encoding='utf8') as f:
            lines = f.read().split('\n')

        pinyin_list = lines[1].split(" ")

        return pinyin_list

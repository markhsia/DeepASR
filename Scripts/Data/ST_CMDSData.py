import os
import json
from .AcousticData import AcousticData

class ST_CMDSData(AcousticData):
    def _get_one_text(self, audio_fp):
        text_fp = audio_fp[:-3] + 'txt'
        with open(text_fp,'r') as f:
            text = f.read().strip()
        return text
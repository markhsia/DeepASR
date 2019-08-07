import os
import json
from .AcousticData import AcousticData

# aidatatang_transcript_fp = "/data/speech/Aidatatang/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt"

# def _read_transcript(transcript_fp):
#     with open(transcript_fp, 'r') as f:
#         transcript_list = f.read().split('\n')
#     file2text_dict = {}
#     for tran in transcript_list:
#         file_name = tran[:file_len]+'.wav'
#         text = tran[file_len:]
#         file2text_dict[file_name] = text.replace(' ', '')

#     return file2text_dict


class AiDataTang(AcousticData):
    def _get_one_text(self, audio_fp):
        text_fp = audio_fp[:-3] + 'txt'
        with open(text_fp,'r') as f:
            text = f.read().strip()
        return text


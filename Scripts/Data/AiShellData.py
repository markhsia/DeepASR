import os
import json
from .AcousticData import AcousticData

file_len = len('BAC009S0002W0122')

aishell_transcript_fp = "/data/speech/AiShell/data_aishell/transcript/aishell_transcript_v0.9.txt"

def _read_transcript(transcript_fp):
    with open(transcript_fp, 'r') as f:
        transcript_list = f.read().split('\n')
    file2text_dict = {}
    for tran in transcript_list:
        file_name = tran[:file_len]+'.wav'
        text = tran[file_len:]
        file2text_dict[file_name] = text.replace(' ', '')

    return file2text_dict

class AiShellData(AcousticData):
    file2text_dict = _read_transcript(aishell_transcript_fp)

    def _get_one_text(self, audio_fp):
        basename = os.path.basename(audio_fp)
        return self.file2text_dict[basename]


import os
import json
from .AcousticData import AcousticData


set1_transcript_fp = "/data/speech/Primewords/primewords_md_2018_set1/set1_transcript.json"

def _read_transcript(transcript_fp):
    with open(transcript_fp, 'r') as f:
        transcript_list = json.load(f)
    file2text_dict = {}
    for trans in transcript_list:
        file2text_dict[trans['file']] = trans['text'].replace(' ', '')

    return file2text_dict

class PrimewordsData(AcousticData):
    file2text_dict = _read_transcript(set1_transcript_fp)

    def _get_one_text(self, audio_fp):
        basename = os.path.basename(audio_fp)
        return self.file2text_dict[basename]

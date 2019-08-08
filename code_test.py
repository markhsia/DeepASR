
from Scripts.Data.AiShellData import AiShellData
from Scripts.Data.AiDataTang import AiDataTang
from Scripts.Data.PrimewordsData import PrimewordsData
from Scripts.Data.Thchs30Data import Thchs30Data
from Scripts.Data.ST_CMDSData import ST_CMDSData
from Scripts.Data.MagicData import MagicData

def test_DataOBJ(DataOBJ,example_fp):
    dataObj = DataOBJ(filepath = example_fp)
    print("测试%s:"%(dataObj.__class__.__name__))
    print(dataObj._get_one_text(example_fp))
    print(dataObj._get_one_pinyin_list(example_fp))
    print(dataObj.get_label())

if __name__ == '__main__':
    for DataOBJ,example_fp in (
            (AiShellData,"/data/speech/AiShell/data_aishell/wav/train/S0002/BAC009S0002W0122.wav"),
            (AiDataTang,"/data/speech/Aidatatang/aidatatang_200zh/corpus/train/G0013/T0055G0013S0001.wav"),
            (PrimewordsData,"/data/speech/Primewords/primewords_md_2018_set1/audio_files/0/00/000017e3-c450-4ec6-ae77-d1e8a6a57897.wav"),
            (Thchs30Data,"/data/speech/thchs30/openslr_format/data_thchs30/train/A11_0.wav"),
            (ST_CMDSData,"/data/speech/ST-CMDS/ST-CMDS-20170001_1-OS/20170001P00191A0008.wav"),
            (MagicData,"/data/speech/MAGICDATA/dev/38_5731/38_5731_20170915091709.wav")
        ):
        test_DataOBJ(DataOBJ,example_fp)

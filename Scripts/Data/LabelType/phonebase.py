import json
import os
SAVE_DIR = os.path.dirname(__file__)

py2phones_dict_fp = os.path.join(SAVE_DIR, 'py2phones_dict.json')
phone2num_sdict_fp = os.path.join(SAVE_DIR, 'phone2num_sdict.json')
phone2num_dict_fp = os.path.join(SAVE_DIR, 'phone2num_dict.json')
PinYinTable_fp = os.path.join(SAVE_DIR, 'PinYinTable.csv')

def prepare_phonebase():
    tables = []
    with open(PinYinTable_fp, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        for line in lines:
            tables.append(line.split(','))

    py2phones_dict = {}
    for i in range(1, len(tables)):
        for j in range(1, len(tables[0])):
            if len(tables[i][j]):
                py2phones_dict[tables[i][j]] = (tables[0][j], tables[i][0])

    phone2num_sdict = {
        "consonant": {},
        "vowel": {},
        "silence": {}
    }
    phone2num_dict = {}
    k = 0
    for c in tables[0][1:]:
        phone2num_sdict["consonant"][c] = k
        phone2num_dict[c] = k
        k += 1

    for i in range(1, len(tables)):
        v = tables[i][0]
        for t in (1, 2, 3, 4, 5):
            phone2num_sdict["vowel"][v + str(t)] = k
            phone2num_dict[v + str(t)] = k
            k += 1
    # 最后一个设置为静音
    sil = 'sil'
    phone2num_sdict['silence'][sil] = k
    phone2num_dict[sil] = k

    with open(py2phones_dict_fp, 'w', encoding='utf8') as f:
        print("保存py2phones_dict,共%d个拼音（不包含声调）" % len(py2phones_dict))
        json.dump(py2phones_dict, f)

    with open(phone2num_sdict_fp, 'w', encoding='utf8') as f:
        print("保存phone2num_sdict,共%d个辅音，%dx5个元音（每个包含5声调）%d个静音" % (len(phone2num_sdict['consonant']), len(
            phone2num_sdict['vowel'])/5, len(phone2num_sdict['silence'])))
        json.dump(phone2num_sdict, f)
    with open(phone2num_dict_fp, 'w', encoding='utf8') as f:
        print("保存phone2num_dict,共%d个音" % (len(phone2num_dict)))
        json.dump(phone2num_dict, f)

for p in (py2phones_dict_fp, phone2num_dict_fp):
    if not os.path.exists(p):
        print("缺少%s文件，准备重新生成" % p)
        prepare_phonebase()
        break

with open(py2phones_dict_fp, 'r', encoding='utf8') as f:
    # 【拼音】转【音素】用
    py2phones_dict = json.load(f)
    print("读取py2phones_dict,共%d个拼音（不包含声调）" % len(py2phones_dict))


with open(phone2num_dict_fp, 'r', encoding='utf8') as f:
    # 给【音素】编码用
    phone2num_dict = json.load(f)
    print("读取phone2num_dict,共%d个音" % (len(phone2num_dict)))
phone_NUM = len(phone2num_dict)
num2phone_dict = dict([(v, k) for (k, v) in phone2num_dict.items()])

assert(max([num for num in phone2num_dict.values()])+1 == len(phone2num_dict))


def phone2num(phone_list):
    # 【音素】符号转成数字编号
    return [phone2num_dict[pho] for pho in phone_list]


def num2phone(phoneNUM_list):
    # 数字编号转成【音素】符号
    return [num2phone_dict[num] for num in phoneNUM_list]


def pinyin_list2phone_list(pinyin_list):
    # 分解【拼音】得到【音素】列表phone_list
    phone_list = []
    for py in pinyin_list:
        tone = py[-1]
        if py[:-1] in py2phones_dict:
            phones = py2phones_dict[py[:-1]]
        else:
            raise Exception("%s未包含与拼音表中，请检查拼音是否正确" % py)

        if phones[0] == 'None':
            tone_phones = [phones[1] + tone]
        else:
            tone_phones = [phones[0], phones[1] + tone]
        phone_list.extend(tone_phones)

    return phone_list

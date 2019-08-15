from pypinyin import pinyin, Style,load_single_dict
import json
import os

SAVE_DIR = os.path.dirname(__file__)
pinyin2num_dict_fp = os.path.join(SAVE_DIR, 'pinyin2num_dict.json') # 拼音-->数字 字典。
PinYinTable_fp = os.path.join(SAVE_DIR, 'PinYinTable_modern.csv')# 拼音与音素原始表，若不存在上述文件，则根据此表生成

if os.path.basename(PinYinTable_fp) == 'PinYinTable_classic.csv':
    load_single_dict({
        ord('嗯'):'en2',
        ord('哟'):'you4',
        })
else:
    load_single_dict({
        ord('嗯'):'ng2',
        })
def prepare_pinyinbase():
    tables = []
    with open(PinYinTable_fp, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')
        for line in lines:
            tables.append(line.split(','))

    pinyin2num_dict = {}
    k = 0
    for i in range(1, len(tables)):
        for j in range(1, len(tables[0])):
            if len(tables[i][j]):
                for t in (1, 2, 3, 4, 5):
                    pinyin2num_dict[tables[i][j]+str(t)] = k
                    k += 1
    # 最后一个设置为静音
    sil = 'sil'
    pinyin2num_dict[sil] = k

    with open(pinyin2num_dict_fp, 'w', encoding='utf8') as f:
        print("保存pinyin2num_dict,共%d个拼音" % (len(pinyin2num_dict)))
        json.dump(pinyin2num_dict, f)


if not os.path.exists(pinyin2num_dict_fp):
    print("缺少%s文件，准备重新生成" % pinyin2num_dict_fp)
    prepare_pinyinbase()

with open(pinyin2num_dict_fp, 'r', encoding='utf8') as f:
    # 给【拼音】编码用
    pinyin2num_dict = json.load(f)
    print("读取pinyin2num_dict,共%d个音" % (len(pinyin2num_dict)))
pinyin_NUM = len(pinyin2num_dict)
num2pinyin_dict = dict([(v, k) for (k, v) in pinyin2num_dict.items()])


def pinyin2num(pinyin_list):
    # 【拼音】符号转成数字编号
    return [pinyin2num_dict[py] for py in pinyin_list]


def num2pinyin(pinyinNUM_list):
    # 数字编号转成【拼音】符号
    return [num2pinyin_dict[num] for num in pinyinNUM_list]

# extra_pinyin_dict = {
#     'A':['ei1'],
#     'B':['bi4'],
#     'C':['sei1'],
#     'D':['di4'],
#     'E':['yi4'],
#     'F':['ai2','fu5'],
#     'G':['ji4'],
#     'H':['ei2','chi5'],
#     'I':['ai1'],
#     'J':['zhei4'],
#     'K':['kei4'],
#     'L':['ai2','ou5'],
#     'M':['ai2','mu5'],
#     'N':['en1'],
#     'O':['ou1'],
#     'P':['pi4'],
#     'Q':['kiu4'],
#     'R':['ar4'],
#     'S':['ai2','si5'],
#     'T':['ti4'],
#     'U':['you1'],
#     'V':['wei1'],
#     'W':['da1','bu5','you5'],
#     'X':['ai2','ke5','si5'],
#     'Y':['wai4'],
#     'Z':['zei4'],
# }

# def deal_errors(chars):
#     pys = []
#     for c in chars:
#         if c in extra_pinyin_dict:
#             pys.extend(extra_pinyin_dict[c])
#     if len(pys) == 0:
#         return None
#     else:
#         return pys

def text2pinyin_list(text):
    pinyin_list = pinyin(text, style=Style.TONE3,errors = 'ignore')
    pinyin_list = list(map(lambda py_l:py_l[0] if py_l[0][-1].isdigit() else py_l[0] + '5',pinyin_list))
    if len(pinyin_list) == 0:
        pinyin_list = ['sil']
    return pinyin_list

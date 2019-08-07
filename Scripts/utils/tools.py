import os
import inspect
import numpy as np
import hashlib


def _md5(string):
    md5 = hashlib.md5()
    md5.update(string.encode('utf-8'))
    return md5.hexdigest()


def joinpath(path, *paths):
    return transferPath(os.path.join(path, *paths))


def abspath(path):
    return (transferPath(os.path.abspath(path)))


def transferPath(path):
    path = os.path.normpath(path)  # 将形如'/a/b/../c/'的路径转变为规范的'/a/c/'
    # path = path.replace('\\', '/')
    return path


def walk_subfiles(paths):
    for path in paths:
        if os.path.isdir(path):
            for cur_dir, _dirs, files in os.walk(path):
                for filename in files:
                    yield os.path.join(cur_dir, filename)
        elif os.path.isfile(path):
            yield path
        else:
            print("%s路径不存在" % (path))



def fresh_print(s):
    pass

import pprint
def format_dict2file(d:dict,filepath:str):
    with open(filepath, 'a') as f:
        print('生成',filepath)
        f.write(pprint.pformat(d, indent=4))
# import re

# def get_replace_func(things2replace):
#     regx = re.compile("[%s]+" %things2replace)
#     def replace_func(string):
#         return regx.sub("",string)
#     return del_func


from Scripts.utils.tools import walk_subfiles
import os
for fp in walk_subfiles(['/data/speech/AiShell/data_aishell/wav/']):
    if fp[-4:] == '.npy':
        print("删除%s"%fp)
        # os.remove(fp)
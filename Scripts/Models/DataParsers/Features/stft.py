import librosa
import numpy as np
def _parse_one_audio_stft(ori_y, ori_sr, fft_s, hop_s, target_sr):
    """
    ori_y, ori_sr:应当由librosa.load读取的原始音频数据ori_y以及音频采样率ori_sr
    fft_s:一个短时傅里叶变换的窗长，单位为秒
    hop_s：窗之间间隔长，单位为秒

    返回：行数为 duration_s(音频长度，秒)//hop_s, 列数为 fft_s * target_sr//2 的二维数组；
        纵向代表时间(每行跨越hop_s秒)，横向代表频率（物理范围：0至(target_sr//2) Hz,每列跨越fft_s Hz），元素大小代表能量（单位：db）
    """
    y = librosa.resample(ori_y, ori_sr, target_sr)
    sr = target_sr

    n_fft = round(fft_s * sr)
    hop_length = round(hop_s * sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=None, window='hann', center=False))
    powerS = S**2
    powerS_db = librosa.power_to_db(powerS)

    return powerS_db.T
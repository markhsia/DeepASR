import librosa
import numpy as np
def _parse_one_audio_melspec(ori_y, ori_sr, fft_s, hop_s, target_sr, n_mels ):
    """
    ori_y, ori_sr:应当由librosa.load读取的原始音频数据ori_y以及音频采样率ori_sr
    fft_s:一个短时傅里叶变换的窗长，单位为秒
    hop_s：窗之间间隔长，单位为秒

    返回：行数为 duration_s(音频长度,秒)//hop_s, 列数为 n_mels 的二维数组，纵向代表时间(一行跨越 hop_s 秒)，横向代表频率（物理范围：0-(target_sr//2) Hz,一列跨越fft_s Hz），大小代表能量（单位：db）
    """
    y = librosa.resample(ori_y, ori_sr, target_sr)
    sr = target_sr

    n_fft = round(fft_s * sr)
    hop_length = round(hop_s * sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=None, window='hann', center=False))
    powerS = S**2
    mel_powerS = librosa.feature.melspectrogram(S=powerS,n_mels = n_mels)
    mel_powerS_db = librosa.power_to_db(mel_powerS)

    return mel_powerS_db.T
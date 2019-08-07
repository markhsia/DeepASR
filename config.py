# 文件夹配置
model_save_dir = 'saved_models'
cache_dir = '/data/A/cache'

# 数据配置

'''一个拼音最短大概0.1s,
若hop_s为0.016s,那么6.25个hop覆盖0.1s'''
stft_fea = {
    'name':'stft',
    'kwargs':{
        'fft_s': 0.128,  # fft_s:一个短时傅里叶变换的窗长，单位为秒
        'hop_s': 0.016,  # hop_s：窗之间间隔长，单位为秒
        'target_sr': 8000, # 统一音频采样率目标，音频将自动重采样为该采样率
    }
}

mel_fea = {
    'name':'mel',
    'kwargs':{
        'fft_s': 0.128,  # fft_s:一个短时傅里叶变换的窗长，单位为秒
        'hop_s': 0.016,  # hop_s：窗之间间隔长，单位为秒
        'target_sr': 8000, # 统一音频采样率目标，音频将自动重采样为该采样率
        'n_mels': 128 # mel 特征维度
    }
}
feature = stft_fea
label_type = 'pinyin'

#训练配置
epochs = 500
batch_size = 256
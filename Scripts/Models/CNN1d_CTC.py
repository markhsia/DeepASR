from .AcousticModel import AcousticModel
    
class CNN1d_CTC_PinYin_Sample(AcousticModel):
    '''(AiShell)(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=19)(loss=9.1)(val_loss=12.6)
        (AiShell)(gpu_n=1)(feature_name=mel)(label_type=pinyin)/final(epoch=119)(loss=2.4)(val_loss=18.0)
    '''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_PinYin_Sample_Regular(AcousticModel):
    '''(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=74)(loss=10)(val_loss=17).keras_weight
        (AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/final(epoch=500)(loss=9)(val_loss=18).keras_weight
    '''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        from keras import regularizers
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, 5, padding="same",activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_PinYin_Sample_lessDropout(AcousticModel):
    '''saved_models/CNN1d_CTC_PinYin_Sample_lessDropout(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=177)(loss=6.4)(val_loss=8.8).keras_weight
    saved_models/CNN1d_CTC_PinYin_Sample_lessDropout(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/final(epoch=277)(loss=6.1)(val_loss=9.3).keras_weight
    '''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.2)(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.2)(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_PinYin_Sample_Dropout(AcousticModel):
    '''saved_models/CNN1d_CTC_PinYin_Sample_Dropout(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)_1/best_val_loss(epoch=258)(loss=14)(val_loss=11).keras_weight
    saved_models/CNN1d_CTC_PinYin_Sample_Dropout(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)_1/final(epoch=700)(loss=13)(val_loss=12).keras_weight'''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.3)(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_PinYin_Sample_Dropout2(AcousticModel):
    '''比上面网络做大了些,不会过拟合,loss 10
    saved_models/CNN1d_CTC_PinYin_Sample_Dropout2(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)_1/best_val_loss(epoch=239)(loss=10)(val_loss=9).keras_weight
    saved_models/CNN1d_CTC_PinYin_Sample_Dropout2(AiShell)/(gpu_n=1)(feature_name=mel)(label_type=pinyin)_1/final(epoch=260)(loss=10)(val_loss=9).keras_weight'''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(320, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(512, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.3)(x)
        x = Conv1D(768, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Conv1D(512, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_PinYin_RealSample(AcousticModel):
    '''尽量简化网络,效果不好'''
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=128,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(128, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(128, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

class CNN1d_CTC_Phone_RealSample(AcousticModel):
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'phone'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        input_data = Input(name=input_layername, shape=(None,self.DataParser.AUDIO_FEATURE_LENGTH))

        x = Conv1D(filters=96,kernel_size=7, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(128, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out

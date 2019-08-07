from .AcousticModel import AcousticModel


class CNN2d_CTC_Phone(AcousticModel):
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'phone'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Input, Reshape  # , Flatten
        from keras.layers import TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
        from keras import backend as K
        from keras.optimizers import SGD, Adadelta, Adam
        
        input_data = Input(name=input_layername, shape=(None, self.DataParser.AUDIO_FEATURE_LENGTH, 1))

        x = Conv2D(32, (3, 3), use_bias=False, activation='relu',padding='same',kernel_initializer='he_normal')(input_data)  # 卷积层
        x = MaxPooling2D(pool_size=(2, 2), strides=None,padding="valid")(x)  # 池化层
        x = Conv2D(64, (3, 3), use_bias=True,activation='relu', padding='same',kernel_initializer='he_normal')(x)  # 卷积层
        x = MaxPooling2D(pool_size=(1, 2), strides=None,padding="valid")(x)  # 池化层
        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(x)  # 卷积层
        x = MaxPooling2D(pool_size=(1, 2), strides=None,padding="valid")(x)  # 池化层
        x = Conv2D(128, (3, 3), use_bias=True,activation='relu', padding='same',kernel_initializer='he_normal')(x)  # 卷积层
        x = MaxPooling2D(pool_size=(1, 2), strides=None,padding="valid")(x)  # 池化层
        box_shape = x._keras_shape
        xr = Reshape((-1, box_shape[2]*box_shape[3]))(x)  # Reshape层
        xr = TimeDistributed(Dense(256, activation="relu",use_bias=True, kernel_initializer='he_normal'))(xr)
        xr = Dense(self.LabelParser.LABEL_NUM, use_bias=True,kernel_initializer='he_normal')(xr)  # 全连接层
        softmax_out = Activation('softmax', name='softmax_out')(xr)
        
        return input_data,softmax_out
    
class CNN2d_CTC_PinYin(AcousticModel):
    def main_structure(self,input_layername):
        '''input_keras_layer -> softmax_out_layer'''   
        assert self.LabelParser.label_type == 'pinyin'
        
        from keras.models import Sequential, Model
        from keras.layers import Dense, Input, Reshape,BatchNormalization,ReLU,Dropout  # , Flatten
        from keras.layers import TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
        from keras import backend as K
        from keras.optimizers import SGD, Adadelta, Adam
        from keras import backend as K
        import tensorflow as tf
         
        input_data = Input(name=input_layername, shape=(None, self.DataParser.AUDIO_FEATURE_LENGTH, 1))
        
        layer_ = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',dilation_rate=(2, 2))(input_data)  # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=(1,2), strides=None, padding="valid")(layer_)  # 池化层

        layer_ = Conv2D(32, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal',dilation_rate=(2, 2))(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层

        layer_ = Conv2D(64, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=(1,2), strides=None, padding="valid")(layer_)  # 池化层

        layer_ = Conv2D(64, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层

        layer_ = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_)  # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=(1,2), strides=None, padding="valid")(layer_)  # 池化层

        layer_ = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_)  # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = Conv2D(128, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_) # 池化层

        layer_ = Conv2D(256, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_)  # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=(1, 2), strides=None, padding="valid")(layer_)  # 池化层

        layer_ = Conv2D(256, (3,3), use_bias=True, padding='same', kernel_initializer='he_normal')(layer_) # 卷积层
        layer_ = BatchNormalization()(layer_)
        layer_ = ReLU()(layer_)
        layer_ = MaxPooling2D(pool_size=(1,2), strides=None, padding="valid")(layer_)  # 池化层

        layer_ = Reshape((200, 256))(layer_) #Reshape层

        layer_ = Dropout(0.5)(layer_)
        layer_ = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_) # 全连接层
        layer_ = BatchNormalization()(layer_)
        layer_ = Dense(self.LabelParser.LABEL_NUM, use_bias=True, kernel_initializer='he_normal')(layer_) # 全连接层
        layer_ = BatchNormalization()(layer_)
        softmax_out = Activation('softmax', name='softmax_out')(layer_)
        
        return input_data,softmax_out

import os, math, json, shutil, traceback
from keras.utils.generic_utils import unpack_singleton, to_list
from keras.utils import Sequence,print_summary
from tqdm import tqdm
import numpy as np
from abc import abstractmethod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

from .DataParsers.BaseParser import BaseDataParser, BaseLabelParser
from ..Data.BaseData import BaseData as DataObj
from ..utils.others import ERROR_DATA_INFO
error_info = ERROR_DATA_INFO('data_error_infos_in_fitting.jsons')

PADDED_DATAS_NAME = 'padded_datas'
PADDED_LABELS_NAME = 'padded_labels'
DATA_LENGTHS_NAME = 'data_lengths'
LABEL_LENGHTS_NAME = 'label_lengths'
CTC_LOSS_NAME = 'CTC_loss'

class ShorterThanLabelError(Exception):
    pass

class DataGen4KerasCTC(Sequence):
    def __init__(self, 
                ordered_DataObj_gen, 
                batch_size:int, 
                input_data_shape:tuple,
                DataParser: BaseDataParser, 
                LabelParser: BaseDataParser,
                ignore_error:bool = True,
                save_error_info:bool = True):
        self.ordered_DataObj_gen = ordered_DataObj_gen
        self.batch_size = batch_size
        self.input_data_shape = input_data_shape
        self.DataParser = DataParser
        self.LabelParser = LabelParser

        self.ignore_error = ignore_error
        self.save_error_info = save_error_info
        # self.ERROR_INFOS = [] if self.save_error_info else None


    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        data_list = []
        label_list = []
        for data_obj in self.ordered_DataObj_gen[
                index*self.batch_size:
                min((index+1)*self.batch_size, len(self.ordered_DataObj_gen))
            ]:
            data,label = None,None
            try:
                data = self.DataParser(data_obj)
                label = self.LabelParser(data_obj)
                if data.shape[0]//8 < len(label):
                    raise ShorterThanLabelError("音频长度%d//8 小于 标签长度%d"%(data.shape[0],len(label)))
            except Exception:
                print("\n=================\n出错data的id为:",data_obj.id)
                if self.save_error_info:
                    error_info.save_error_info(data_obj)
                if self.ignore_error:
                    print("\t****错误信息****")
                    print("\t"+traceback.format_exc().replace('\n','\n\t'))
                    print("\n=================\n")
                    continue
                else:
                    raise
            data_list.append(data)
            label_list.append(label)
        padded_datas, data_lengths = self._parse_datas4ctc(data_list,self.input_data_shape)
        padded_labels, label_lengths = self._parse_labels4ctc(label_list)

        inputs = {
            PADDED_DATAS_NAME: padded_datas,
            PADDED_LABELS_NAME: padded_labels,
            DATA_LENGTHS_NAME: data_lengths,
            LABEL_LENGHTS_NAME: label_lengths,
        }
        outputs = {CTC_LOSS_NAME: np.zeros(padded_datas.shape[0])}
        return inputs, outputs

    def __len__(self):
        return int(np.ceil(len(self.ordered_DataObj_gen) / float(self.batch_size)))

    @staticmethod
    def _parse_datas4ctc(datas_list,input_data_shape):
        data_lengths = np.array([data.shape[0]
                                 for data in datas_list], ndmin=2).T
        padded_data_shape = tuple([len(datas_list), data_lengths.max()] + list(input_data_shape[2:]))
        padded_datas = np.zeros(padded_data_shape)
        if padded_datas.ndim == datas_list[0].ndim + 2:
            for i, data in enumerate(datas_list):
                padded_datas[i, : data.shape[0],:] = np.expand_dims(data,axis = -1)
        elif padded_datas.ndim == datas_list[0].ndim + 1:
            for i, data in enumerate(datas_list):
                padded_datas[i, : data.shape[0]] = data  
        else:
            raise Exception("Data Dhape ERROR, padded_datas.ndim:%d, while data.ndim:%d"%(padded_datas.ndim,datas_list[0].ndim))
        return padded_datas, data_lengths

    @staticmethod
    def _parse_labels4ctc(labels_list):
        label_lengths = np.array([len(label)
                                  for label in labels_list], ndmin=2).T
        padded_labels = np.zeros(
            (len(labels_list), label_lengths.max()))
        for i, label in enumerate(labels_list):
            padded_labels[i, : len(label)] = label

        return padded_labels, label_lengths


class KerasCTCBaseModel():
    def __init__(self, name:str, DataParser: BaseDataParser, LabelParser: BaseLabelParser, save_dir='saved_model'):
        self.name = name
        self.DataParser = DataParser
        self.LabelParser = LabelParser
        self.env_pred = False
        self.model, self.predict_model = self.create_model()
        self._pre_tf_env()
        assert self.model is not None, "模型初始化失败"
        self.input_data_shape = self.model.get_layer(PADDED_DATAS_NAME).input_shape
        self.main_save_dir = save_dir
        self.cur_save_dir = None
        self.load_weight_path = None

        # 解决服务配置问题
        self.predict_model._make_predict_function()

    @abstractmethod
    def main_structure(self, input_layername):
        '''input_keras_layer -> softmax_out_layer'''

        raise NotImplementedError

    # def edit_distance()
    # tf.edit_distance

    def _capture_summary(self,s):
        self.model_summary += (s+'\n')
        print(s)

    def create_model(self):
        # from keras import Model,Input,
        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization  # , Flatten
        from keras.layers import Lambda, TimeDistributed, Activation, Conv2D, MaxPooling2D  # , Merge
        from keras import backend as K


        def ctc_lambda_func(inputs):
            softmax_out, label, data_lengths, label_lenghts = inputs
            max_data_len = K.max(data_lengths)
            max_out_len = K.cast(K.shape(softmax_out)[1],dtype = data_lengths.dtype)
            out_lengths = K.round(data_lengths*max_out_len/max_data_len)
            return K.ctc_batch_cost(y_true=label, y_pred=softmax_out, input_length=out_lengths, label_length=label_lenghts)

        label = Input(name=PADDED_LABELS_NAME, shape=[None], dtype='float32')
        data_lengths = Input(name=DATA_LENGTHS_NAME, shape=[1], dtype='int64')
        label_lenghts = Input(name=LABEL_LENGHTS_NAME,
                              shape=[1], dtype='int64')

        input_data, softmax_out = self.main_structure(
            input_layername=PADDED_DATAS_NAME)
        predict_model = Model(input_data, softmax_out)

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name=CTC_LOSS_NAME)(
            [softmax_out, label, data_lengths, label_lenghts])

        model = Model([input_data, label, data_lengths,
                       label_lenghts], loss_out)

        self.model_summary = ''
        print_summary(model,print_fn=self._capture_summary)
        # model.summary()

        return model, predict_model
    
    def _pre_tf_env(self):
        if not self.env_pred:
            import tensorflow as tf
            import keras.backend.tensorflow_backend as KTF
            # 指定第?块GPU可用
            print(os.popen('nvidia-smi').read())
            which_GPU = input('which_GPU?')
            os.environ["CUDA_VISIBLE_DEVICES"] = which_GPU
            CP = tf.ConfigProto(allow_soft_placement = True,
                                log_device_placement = False)
            CP.gpu_options.allow_growth = True  #不全部占满显存, 按需分配
            # CP.gpu_options.per_process_gpu_memory_fraction = 0.9
            sess = tf.Session(config=CP)
            KTF.set_session(sess)
            
            from keras.utils import multi_gpu_model
            self.gpu_num = os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1
            if self.gpu_num >= 2:
                print("使用数据并行方法在%d个GPU上运行"%self.gpu_num)
                self.model = multi_gpu_model(self.model,gpus=self.gpu_num,cpu_relocation=True)              

            # clipnorm seems to speeds up convergence
            #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
            #ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
            from keras.optimizers import SGD, Adadelta, Adam
            opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                    decay=0.0, epsilon=10e-8)
            self.model.compile(
                loss={'CTC_loss': lambda _label,out: out},
                optimizer=opt,
                 )
            KTF.get_session().run(tf.global_variables_initializer())

            self.env_pred = True
        print("使用CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    def _batch_gen4DataObjIter(self,DataObjIter,batch_size):
        return DataGen4KerasCTC(
                        DataObjIter, 
                        batch_size,
                        self.input_data_shape,
                        self.DataParser,
                        self.LabelParser
                    )

    @abstractmethod
    def _create_fit_infos_dict(self,data_name,train_DataObjIter,dev_DataObjIter,batch_size,epochs):
        fit_infos_dict = {
            'gpu_n':self.gpu_num,
            'data_name': data_name,
            'train_data_num':len(train_DataObjIter),
            'dev_data_num':len(dev_DataObjIter),
            'ModelOBJ_name':self.__class__.__name__,
            'model_name':self.name,
            'max_epochs':epochs,
            'batch_size':batch_size,
            'load_weight_path':self.load_weight_path,
            'main_save_dir':self.main_save_dir,
        }

        abstract_infos_dict = {
            'gpu_n':self.gpu_num,
        }
        return fit_infos_dict,abstract_infos_dict
    
    def _prepare_fit(self,data_name,train_DataObjIter,dev_DataObjIter,batch_size,epochs):

        fit_infos_dict,abstract_infos_dict = self._create_fit_infos_dict(data_name,train_DataObjIter,dev_DataObjIter,batch_size,epochs)

        sub_dir = data_name
        attr_sub_dir = ''
        for k,v in abstract_infos_dict.items():
            attr_sub_dir += '(%s=%s)'%(k,v)
        cur_model_savedir = os.path.join(self.main_save_dir,self.name,sub_dir,attr_sub_dir)
        base_len = len(cur_model_savedir)
        _i = 1
        while os.path.exists(cur_model_savedir):
            cur_model_savedir = cur_model_savedir[:base_len] + '_%d'%_i
            _i += 1
        
        fit_infos_dict['cur_model_savedir'] = cur_model_savedir

        info_str = '=====模型信息=====\n'
        for k,v in fit_infos_dict.items():
            info_str += "%s:%s\n"%(k,v)
        info_str += '=================\n'
        print(info_str)
        input("请确认模型信息,按任意键继续...")
        print("创建目录:",cur_model_savedir)
        os.makedirs(cur_model_savedir)
        with open(os.path.join(cur_model_savedir,"infos.txt"),'a') as f:
            print("写入infos.txt")
            f.write(info_str)
            f.write(self.model_summary)
        
        
        # print("@模型保存主文件夹:%s\n\t当前模型保存文件夹:%s"%(self.main_save_dir,cur_model_savedir))
        # print("@训练集数据数:",len(train_DataObjIter))
        # print("@验证机数据数:",len(dev_DataObjIter))

        self.cur_save_dir =  cur_model_savedir

    def fit(self,data_name:str, train_DataObjIter, dev_DataObjIter, batch_size, epochs,patience = 20):
        self._prepare_fit(data_name,train_DataObjIter,dev_DataObjIter,batch_size,epochs)

        from keras.callbacks import EarlyStopping, ModelCheckpoint

        # y_pred_TIMESIZE = ctc_model.get_layer('y_pred').output_shape[1]

        train_batch_gen = self._batch_gen4DataObjIter(train_DataObjIter,batch_size)
        validation_batch_gen = self._batch_gen4DataObjIter(dev_DataObjIter,batch_size)

        # 坑爹啊，模型用多GPU训练的话，载入模型权重也要同样多GPU载入
        tail_name = '(epoch={epoch})(loss={loss:.1f})(val_loss={val_loss:.1f}).keras'
        best_val_loss_model_savepath = os.path.join(self.cur_save_dir,'best_val_loss'+tail_name)
        final_model_savepath = os.path.join(self.cur_save_dir,'final'+tail_name)
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
            ),
            MyModelCheckpoint(
                filepath=best_val_loss_model_savepath+'_weight',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        # try:
        history = self.model.fit_generator(
            train_batch_gen,
            epochs=epochs,
            validation_data=validation_batch_gen,
            callbacks=callbacks_list,
            max_queue_size=1024,
            workers=32,
            use_multiprocessing=False,
            shuffle = True)
        self.plot_history(history, final_model_savepath)
        # except:
        #     traceback.print_exc()
        #     print("训练中断！！！")
        # myfit_generator(self.model,train_batch_gen,epochs,batch_size,validation_batch_gen)
        
        # val_loss = history.history['val_loss'][-1]
        # try:
        #     self.model.save(model_savepath.format(epoch = epochs,val_loss = val_loss)+'_model.final')
        # except:
        #     print("save_final_model失败!!!")

        try:
            self.model.save_weights(final_model_savepath.format(epoch = len(history.history['loss']),loss =history.history['loss'][-1],val_loss = history.history['val_loss'][-1])+'_weight')
        except:
            print("save_final_model_weight失败!!!")
            traceback.print_exc()
    def test(self, test_DataObjIter, batch_size):
        test_batch_gen = self._batch_gen4DataObjIter(test_DataObjIter,batch_size)
        print('验证（共%d个数据)' % (len(test_DataObjIter)))
        print('验证loss:', self.model.evaluate_generator(generator=test_batch_gen))

    def predict(self,data_obj: DataObj):
        # padded_datas,data_lengths = DataGen4KerasCTC._parse_datas4ctc([self.DataParser(data_obj)])
        data_batch_gen = self._batch_gen4DataObjIter([data_obj],batch_size = 1)
        for inputs,_ in data_batch_gen:
            softmax_out = self.predict_model.predict(inputs[PADDED_DATAS_NAME])
            
            encoded_label = self._keras_ctc_decode(softmax_out,inputs[DATA_LENGTHS_NAME])
            decoded_label = self.LabelParser.decode_label(encoded_label)
            return decoded_label
    
    @staticmethod
    def _keras_ctc_decode(softmax_out,data_lengths):
        max_out_len = softmax_out.shape[1]
        max_data_len = data_lengths.max()
        out_lenghts = np.squeeze(np.round(data_lengths*max_out_len/max_data_len),axis=-1)
        from keras import backend as K
        res = K.ctc_decode(y_pred=softmax_out, input_length=out_lenghts,
                           greedy=True, beam_width=100, top_paths=1)
        decoded_pred = K.get_value(res[0][0])[0]
        return decoded_pred
    
    def load_weight(self, load_weight_path):
        self.load_weight_path = load_weight_path
        self.model.load_weights(load_weight_path,by_name=True)

    @staticmethod
    def plot_history(history, model_savepath):

        # acc = history.history['acc']
        # val_acc = history.history['val_acc']

        losses = history.history['loss']
        val_losses = history.history['val_loss']
        epoch_range = range(1, len(losses) + 1)
        model_savepath = model_savepath.format(epoch = len(losses) + 1,loss =losses[-1], val_loss =val_losses[-1])
        plt.figure()
        # plt.subplot(121)
        # plt.plot(epochs, acc, 'bo', label='Training acc')
        # plt.plot(epochs, val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Acc')
        # plt.legend()
        # plt.subplot(122)
        plt.plot(epoch_range, losses, 'bo', label='Training loss')
        plt.plot(epoch_range, val_losses, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('%s.png' % (model_savepath))

from keras.callbacks import Callback,warnings
class MyModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.last_filepath = None
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.last_filepath is not None:
                            os.remove(self.last_filepath)
                        self.last_filepath = filepath
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
def _get_batchsize(x):
    if x is None or len(x) == 0:
        # Handle data tensors support when no input given
        # step-size = 1 for data tensors
        batch_size = 1
    elif isinstance(x, list):
        batch_size = x[0].shape[0]
    elif isinstance(x, dict):
        batch_size = list(x.values())[0].shape[0]
    else:
        batch_size = x.shape[0]
    if batch_size == 0:
        raise ValueError('Received an empty batch. '
                         'Batches should contain '
                         'at least one item.')
    return batch_size


def myfit_generator(model, train_batch_gen, epochs: int, batchsize: int, validation_batch_gen=None):
    def avg_loss(losses, batch_sizes):
        averages = []
        for i in range(len(losses[0])):
            if i not in stateful_metric_indices:
                averages.append(np.average([out[i] for out in losses],
                                           weights=batch_sizes))
            else:
                averages.append(np.float64(losses[-1][i]))
        return unpack_singleton(averages)
    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]
    print('out_labels:', out_labels, 'callback_metrics:', callback_metrics)

    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    for epoch in range(epochs):
        # train
        train_loss = None
        trained_data_num = 0
        train_losses = []
        train_batch_sizes = []
        for cur_batch_x, cur_batch_y in tqdm(train_batch_gen, desc='Training... epoch:{} trained_data:{} train_loss:{}'.format(epoch, trained_data_num, train_loss)):
            # self.model.fit(cur_batch_x,cur_batch_y,batch_size=batch_size,epochs=1,initial_epoch=epoch)

            train_loss = model.train_on_batch(cur_batch_x, cur_batch_y)

            train_loss = to_list(train_loss)

            batch_size = _get_batchsize(cur_batch_x)
            trained_data_num += batch_size

            train_loss = to_list(train_loss)
            train_losses.append(train_loss)

            train_batch_sizes.append(batch_size)

        print("fit_loss:", avg_loss(train_losses, train_batch_sizes))
        # evaluate

        if validation_batch_gen is None:
            continue

        evaluate_loss = None
        evaluated_data_num = 0
        val_losses = []
        val_batch_sizes = []
        for cur_devbatch_x, cur_devbatch_y in tqdm(validation_batch_gen, desc='Validating... evaluated_data:{} evaluate_loss:{}'.format(evaluated_data_num, evaluate_loss)):
            evaluate_loss = model.test_on_batch(cur_devbatch_x, cur_devbatch_y)
            batch_size = _get_batchsize(cur_devbatch_x)
            evaluated_data_num += batch_size

            evaluate_loss = to_list(evaluate_loss)
            val_losses.append(evaluate_loss)

            val_batch_sizes.append(batch_size)

        print("val_loss:", avg_loss(val_losses, val_batch_sizes))

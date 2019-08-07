from abc import abstractmethod


class BaseData:
    def __init__(self, dataid: str):
        self.id = dataid

    @abstractmethod
    def get_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_label(self):
        raise NotImplementedError

    # def encode_label(self):
    #     assert self.get_data_func is not None,"DataObj.id=%s,该DataObj的encode_label_func未定义"
    #     return self.encode_label_func(self.id)

    # def decode_label(self):
    #     assert self.get_data_func is not None,"DataObj.id=%s,该DataObj的decode_label_func未定义"
    #     return self.decode_label_func(self.id)

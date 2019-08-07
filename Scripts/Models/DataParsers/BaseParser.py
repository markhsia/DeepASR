from abc import abstractmethod
from ...Data.BaseData import BaseData as DataObj
import itertools


class BaseDataParser:
    def __init__(self):
        pass

    def __call__(self, data_obj: DataObj):
        return self.parse_one_data(data_obj)

    @abstractmethod
    def parse_one_data(self, data_obj: DataObj):
        pass


class BaseLabelParser:
    def __init__(self):
        pass

    def __call__(self, data_obj: DataObj):
        return self.parse_one_label(data_obj)

    @abstractmethod
    def parse_one_label(self, data_obj: DataObj):
        pass

    @abstractmethod
    def encode_label(self, label):
        pass

    @abstractmethod
    def decode_label(self, encoded_label):
        pass


# class BaseDataObjIterator:
#     def __init__(self,  ordered=True):
#         self.ordered = ordered
#         self.DataObjs = []

#     def __iter__(self):
#         for data_obj in self.DataObjs:
#             yield data_obj

#     def __len__(self):
#         assert self.ordered, "此数据读取器是无序的"
#         return len(self.DataObjs)

#     def __getitem__(self, index):
#         assert self.ordered, "此数据读取器是无序的"
#         return self.DataObjs[index]

#     def add_DataObjs(self, DataObj_class: DataObj, ids: list([str])):
#         if self.ordered:
#             self.DataObjs.extend(self._make_DataObj_list(DataObj_class, ids))
#         else:
#             self.DataObjs = itertools.chain(
#                 self.DataObjs, self._make_DataObj_gen(DataObj_class, ids))

#     def _make_DataObj_gen(self, DataObj_class: DataObj, ids):
#         for dataid in ids:
#             yield DataObj_class(dataid)

#     def _make_DataObj_list(self, DataObj_class: DataObj, ids):
#         return [data_obj for data_obj in self._make_DataObj_gen(DataObj_class, ids)]

import pickle as pkl
import numpy as np
import time
import random
import os
import sys

class MyDataset:
    def __init__(self):
        # store feature id in document list
        self._documentlist = []
        # store feature value in feature list
        self._featurelist = []
        self._observationlist = []
        self._clicklist = []
        self._conversionlist = []
        self._len = 0
        
    def __len__(self):
        return self._len

    def sample(self, index):
        # index: list
        assert max(index) <= self._len
        sample_documentlist, sample_featurelist, sample_clicklist, sample_conversionlist, sample_observationlist = [], [], [], [], []
        for _index in index:
            sample_documentlist.append(self._documentlist[_index])
            sample_featurelist.append(self._featurelist[_index])
            sample_clicklist.append(self._clicklist[_index])
            sample_conversionlist.append(self._conversionlist[_index])
            sample_observationlist.append(self._observationlist[_index])
        return sample_documentlist, sample_featurelist, sample_clicklist, sample_conversionlist, sample_observationlist
    
    def append(self, document_list, feature_list, click_list, conversion_list, observation_list):
        # append query data
        self._documentlist.append(document_list)
        self._featurelist.append(feature_list)
        self._clicklist.append(click_list)
        self._conversionlist.append(conversion_list)
        self._observationlist.append(observation_list)
        assert len(self._clicklist)==len(self._conversionlist)==len(self._documentlist)==len(self._featurelist)==len(self._observationlist)
        self._len = len(self._clicklist)
    

class DataLoader:
    def __init__(self, data_path, name, save, seq_len, data_type):
        self._name = name
        self._path = data_path
        self._save = save
        self._type = data_type
        self._seqlen = seq_len
    
    def load_data(self):
        if self._name == "taobao":
            return self._load_taobao()
        elif self._name == "ali":
            return self._load_ali()
        elif self._name == "criteo":
            return self._load_criteo()
        else:
            raise NotImplementedError
    
    def _load_taobao(self):
        dataloader = MyDataset()
        if self._type == "train":
            data_name = "taobao/taobao_train_mini.txt"
            file_name = "taobao/taobao_train_mini.txt.pkl"
            pkl_name = "_train.pkl"
        else:
            data_name = "taobao/taobao_test_mini.txt"
            file_name = "taobao/taobao_test_mini.txt.pkl"
            pkl_name = "_test.pkl"
        if os.path.exists(os.path.join(self._path, data_name)):
            _datadir = open(os.path.join(self._path, data_name), 'r')
        else:
            raise NameError
        if os.path.exists(os.path.join(self._path, file_name)):
            _filedir = open(os.path.join(self._path, file_name), 'rb')
            _file = pkl.load(_filedir)
        else:
            raise NameError
        for _seqlen in _file:
            document_list, feature_list = [], []
            click_list, conversion_list, observation_list = [], [], []
            for _d in range(_seqlen):
                _line = _datadir.readline().rstrip().split()
                _split = [_feature.split(":") for _feature in _line[6:]]
                _split = np.reshape(_split, (-1, 3))
                document_list.append(_split[:, 1].astype(np.int).tolist())
                feature_list.append(_split[:, 2].astype(np.float).tolist())
                if int(_line[2]) == 1:
                    # [1, 0] for click
                    click_list.append([1, 0])
                else:
                    click_list.append([0, 1])
                if int(_line[3]) == 1:
                    # [1, 0] for conversion
                    conversion_list.append([1, 0])
                else:
                    conversion_list.append([0, 1])
                observation_list.append([1, 0])
                
                document_list = document_list[-self._seqlen:]
                feature_list = feature_list[-self._seqlen:]
                click_list = click_list[-self._seqlen:]
                conversion_list = conversion_list[-self._seqlen:]
                observation_list = observation_list[-self._seqlen:]
            if _seqlen < self._seqlen:
                for _ in range(_seqlen, self._seqlen):
                    # feature id append 0
                    document_list.append([0])
                    # feature value append 1
                    feature_list.append([1])
                    click_list.append([0, 0])
                    conversion_list.append([0, 0])
                    observation_list.append([0, 0])
            assert len(document_list)==len(feature_list)==len(click_list)==len(conversion_list)==len(observation_list)
            dataloader.append(document_list, feature_list, click_list, conversion_list, observation_list)
        if self._save:
            with open(os.path.join(self._path, self._name+pkl_name), 'wb') as foutfile:
                pkl.dump(dataloader, foutfile)
        _filedir.close()
        _datadir.close()
        return dataloader

    def _load_criteo(self):
        dataloader = MyDataset()
        if self._type == "train":
            data_name = "criteo/criteo_train_mini.txt"
            pkl_name = "_train.pkl"
        else:
            data_name = "criteo/criteo_test_mini.txt"
            pkl_name = "_test.pkl"
        if os.path.exists(os.path.join(self._path, data_name)):
            _datadir = open(os.path.join(self._path, data_name), 'r')
        else:
            raise NameError
        _seqlen, _conversion = [int(_data) for _data in _datadir.readline().rstrip().split()]
        for _ in range(_seqlen):
            # store extra information in document list 
            document_list = []
            # store feature information in feature list
            feature_list = []
            click_list, conversion_list, observation_list = [], [], []
            _line = _data.readline().rstrip().split()
            document_list.append([float(_line[0]), float(_line[1])])
            feature_list.append(_line[1:])
            if _line[1] == 1:
                click_list.append([1, 0])
            else:
                click_list.append([0, 1])
            if _conversion == 1:
                conversion_list.append([1, 0])
            else:
                conversion_list.append([0, 1])
            observation_list.append([1, 0])
            assert len(document_list)==len(feature_list)==len(click_list)==len(conversion_list)==len(observation_list)
            dataloader.append(document_list, feature_list, click_list, conversion_list, observation_list)
        if self._save:
            with open(os.path.join(self._path, self._name+pkl_name), 'wb') as foutfile:
                pkl.dump(dataloader, foutfile)
        _datadir.close()
        return dataloader 

    def _load_ali(self):
        raise NotImplementedError


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Sequence Labeling Model
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SequenceLabelingModel(nn.Module):

    def __init__(self, args):
        """
        Args:
            feature_size_dict: dict, 特征表大小字典
            feature_dim_dict: dict, 输入特征dim字典
            pretrained_embed: np.array, default is None
            dropout_rate: float, dropout rate
        """
        super(SequenceLabelingModel, self).__init__()
        for k, v in args.items():
            self.__setattr__(k, v)

        # feature embedding layer
        self.embedding_dict = dict()
        lstm_input_dim = 0
        for feature_name in self.features:
            embed = nn.Embedding(
                self.feature_size_dict[feature_name], self.feature_dim_dict[feature_name])
            self.embedding_dict[feature_name] = embed
            if feature_name != 'label':
                lstm_input_dim += self.feature_dim_dict[feature_name]
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is not None:
            self.embedding_dict[self.features[0]].weight.data.copy_(torch.from_numpy(self.pretrained_embed))

        # TODO lstm + crf layer
        self.lstm = nn.LSTM(lstm_input_dim, self.lstm_units, num_layers=self.layer_nums, bidirectional=True)

        # dropout layer
        if not hasattr(self, 'dropout_rate'):
            self.__setattr__('dropout_rate', '0.5')
        self.dropout = nn.Dropout(self.dropout_rate)

        # dense layer
        self.hidden2tag = nn.Linear(self.lstm_units*2, self.feature_size_dict['label'])

        self._init_weight()

    def get_tag_score(self, input_dict, batch_size):
        """
        Returns:
            tag_scores: size=[batch_size * max_len, nb_classes]
        """
        # concat inputs
        inputs = []
        for feature_name in self.features:
            inputs.append(self.embedding_dict[feature_name](input_dict[feature_name]))
        inputs = torch.cat(inputs, dim=2)  # size=[batch_size, max_len, input_size]

        inputs = torch.transpose(inputs, 1, 0)  # size=[max_len, batch_size, input_size]

        lstm_output, _ = self.lstm(inputs)
        lstm_output = lstm_output.transpose(1, 0).contiguous()  # [batch_size, max_len, lstm_units]

        # [batch_size * max_len, nb_classes]
        tag_space = self.hidden2tag(lstm_output.view(-1, self.lstm_units*2))

        tag_scores = F.log_softmax(tag_space)

        return tag_scores

    def forward(self, input_dict):
        """
        Args:
            inputs: autograd.Variable, size=[batch_size, max_len]
        """
        batch_size = input_dict[self.features[0]].size()[0]

        return self.get_tag_score(input_dict, batch_size)

    def predict(self, input_dict):
        """
        预测标签
        """
        batch_size = input_dict[self.features[0]].size()[0]
        tag_score = self.get_tag_score(input_dict, batch_size)
        tag_score = tag_score.view(batch_size, self.max_len, -1)  # [batch_size, max_len, -1]
        word_inputs = input_dict[self.features[0]]
        actual_lens = torch.sum(word_inputs>0, dim=1).int()
        _, arg_max = torch.max(tag_score, dim=2)  # [batch_size, max_len]
        tags_list = []
        for i in range(batch_size):
            tags_list.append(arg_max[i].cpu().data.numpy()[:actual_lens.data[i]])

        return tags_list

    def _init_weight(self, scope=.1):
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is None:
            self.embedding_dict[self.features[0]].weight.data.uniform_(-scope, scope)
            if self.use_cuda:
                self.embedding_dict[self.features[0]].weight.data = \
                    self.embedding_dict[self.features[0]].weight.data.cuda()
        for feature_name in self.features[1:]:
            self.embedding_dict[feature_name].weight.data.uniform_(-scope, scope)
            if self.use_cuda:
                self.embedding_dict[feature_name].weight.data = \
                    self.embedding_dict[feature_name].weight.data.cuda()
        self.hidden2tag.weight.data.uniform_(-scope, scope)

    def _init_hidden(self, batch_size):
        h, c = Variable(torch.zeros(self.layer_nums, batch_size, self.lstm_units)), \
            Variable(torch.zeros(self.layer_nums, batch_size, self.lstm_units))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return (h, c)

    def set_use_cuda(self, use_cuda):
        self.use_cuda = use_cuda

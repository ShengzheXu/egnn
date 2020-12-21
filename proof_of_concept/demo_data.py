import os
import sys

from sklearn.mixture import GaussianMixture
from torch.distributions import Normal
import numpy as np
import torch
import torch.optim as optim
import logging
import pandas as pd
from argparse import ArgumentParser
import random

import matplotlib.pyplot as plt

from mdn.models import MixtureDensityNetwork
import torch.optim as optim


class artificial_data_generator(object):
    def __init__(self, weight_list, mean=None):
        self.weight_list = weight_list
        self.mean = mean
        self.df_naive = None
        self.X ,self.y = None, None

    def sample(self, row_num=10000):
        self.df_naive = self._gen_continuous(row_num, self.weight_list)
        return self.df_naive
    
    def agg(self, agg=None):
        if agg is None:
            return None, None
        self.X, self.y = self._agg_window(df_naive, agg)
        return self.X, self.y

    def _gen_continuous(self, row_num, weight_list=[],):
        noise = Normal(0, 1) if self.mean is None else Normal(self.mean, 1)
        row_dep = weight_list[0]

        rt = []
        for i in range(row_num):
            samp = []
            if i == 0:
                samp.append(noise.sample().tolist())
            else:
                samp.append(row_dep*rt[-1][0] + (1-row_dep)*noise.sample().tolist())
            for col_dep_i in weight_list[1:]:
                samp.append(col_dep_i*samp[-1] + (1-col_dep_i)*noise.sample().tolist())
            rt.append(samp)

        df = pd.DataFrame.from_records(rt)
        return df

    def _agg_window(self, df_naive, agg_size):
        col_num = len(df_naive.columns)
        buffer = [[0]*col_num] * agg_size
        X, y = [], []

        list_naive = df_naive.values.tolist()
        for row in list_naive:
            buffer.append(row)
            row_with_window = []
            for r in buffer[-agg_size-1:]:
                row_with_window += r
            X.append(row_with_window)
            y.append(row)

        X = torch.Tensor(X).view(-1, col_num*(agg_size+1))
        y = torch.Tensor(y).view(-1, col_num)
        # print(X)
        # print(y)
        # input()
        return X, y


if __name__ == "__main__":
    # bg = artificial_data_generator(weight_list=[0.9, 0.9])
    # event = artificial_data_generator(weight_list=[0.5, 0.5], mean=2)
    # df_naive = bg.sample(row_num=10)
    # i = 0
    # bk = 0
    # automata = [[0.7, 0.3], [0.5, 0.5]]
    # df = None
    # while i<20:
    #     if bk == 0:
    #         df_naive = bg.sample(row_num=10)
    #         df_naive['color'] = 0
    #     else:
    #         df_naive = event.sample(row_num=5)
    #         df_naive['color'] = 1
    #     if df is None:
    #         df = df_naive
    #     else:
    #         df = pd.concat([df, df_naive], axis= 0)
    #     au = random.uniform(0, 1)
    #     if au >= automata[bk][0]:
    #         bk = 1 - bk
    #     i = i + 1
    # print(df)
    # df = df.reset_index(drop=True)
    # df.to_csv('demo.csv')
    # # df.columns = [0,1,2]
    # plt.plot(df)
    # plt.legend(['dim-0', 'dim-1', 'behavior'])
    # plt.savefig('demo_si.png')
    # plt.clf()

    # df['color+1'] = df['color'].shift(-1)
    # df = df.dropna()
    # print(df)

    df = pd.read_csv('demo.csv')
    df['color+1'] = df['color'].shift(-1)
    df['0+1'] = df['0'].shift(-1)
    df['1+1'] = df['1'].shift(-1)
    df = df.dropna()
    print(df)
    x1 = torch.tensor(df[['0','1']].values.reshape(-1, 2)).float()
    y1 = torch.tensor(df['color+1'].values.reshape(-1, 1)).float()

    model_topic = MixtureDensityNetwork(2, 1, 3)
    # pred_parameters = model(x)
    optimizer = optim.Adam(model_topic.parameters(), lr=0.001)

    for i in range(2001):
        optimizer.zero_grad()
        loss = model_topic.loss(x1, y1).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Iter: %d   Loss: %.2f' % (i, loss.data))


    x2 = torch.tensor(df[['0','1','color+1']].values.reshape(-1, 3)).float()
    y2 = torch.tensor(df[['0+1','1+1']].values.reshape(-1, 2)).float()
    
    model_row = MixtureDensityNetwork(3, 2, 3)
    # pred_parameters = model(x)
    optimizer = optim.Adam(model_row.parameters(), lr=0.001)

    for i in range(2001):
        optimizer.zero_grad()
        loss = model_row.loss(x2, y2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Iter: %d   Loss: %.2f' % (i, loss.data))
    
    # ep = [torch.tensor([.790676, 0.556415]).view(-1, 2).float()]
    # for i in range(200):
    #     x_ = torch.tensor(ep[-1]).view(-1, 2)
        # y_topic_ = model_topic.sample(x_)
    #     y_ = model_row.sample(x_)
        
    #     ep.append(y_)

    # ep2 = torch.cat(ep, dim=1).squeeze().tolist()
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


class mix_type_generator(object):
    def __init__(self, weight_list, attack_prob):
        self.weight_list = weight_list
        self.mean = 0
        self.attack_prob = attack_prob
        self.attack_step = 1
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

    def _attacker(self, step, last_att=None):
        noise = Normal(0, 1)
        if step == 1:
            rt = [2+noise.sample().tolist()]
            rt.append(0.9*rt[0]+noise.sample().tolist())
            rt.append(step)
        else:
            rt = [0.9*last_att[0]+noise.sample().tolist()]
            rt.append(0.9*rt[0]+noise.sample().tolist())
            rt.append(1)
        rt.append(step)
        return rt

    def _gen_continuous(self, row_num, weight_list=[],):
        noise = Normal(0, 1) if self.mean is None else Normal(self.mean, 1)
        row_dep = weight_list[0]

        rt = []
        last_bg = []
        last_att = []
        for i in range(row_num):
            samp = []

            if i == 0:
                samp.append(noise.sample().tolist())
            else:
                samp.append(row_dep*last_bg[0] + (1-row_dep)*noise.sample().tolist())
            for col_dep_i in weight_list[1:]:
                samp.append(col_dep_i*samp[-1] + (1-col_dep_i)*noise.sample().tolist())
            samp.append(0)
            rt.append(samp)
            last_bg = samp.copy()

            # try attack
            au = random.uniform(0, 1)
            if au < self.attack_prob:
                samp = self._attacker(self.attack_step, last_att)
                rt.append(samp)
                last_att = samp.copy()
                self.attack_step = (self.attack_step) % 5 + 1
                while random.uniform(0, 1) < 0.7:
                    samp = self._attacker(self.attack_step, last_att)
                    rt.append(samp)
                    last_att = samp.copy()
                    # self.attack_step = max(self.attack_step + 1, 3)
                    self.attack_step = self.attack_step %5 + 1
                if self.attack_step > 5:
                    self.attack_step = 1

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
    bg = mix_type_generator(weight_list=[0.9, 0.9], attack_prob=0.1)
    df = bg.sample(row_num=50)

    df.to_csv('demo_multistep.csv')
    print(df)
    plt.plot(df)
    plt.legend(['dim-0', 'dim-1', 'behavior', 'mode'])
    plt.savefig('demo_naiveatt.png')
    plt.clf()


    # model = MixtureDensityNetwork(lag, 1, 3)
    # # pred_parameters = model(x)

    # # use this to backprop
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for i in range(100):
    #     optimizer.zero_grad()
    #     loss = model.loss(x, y).mean()
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #     print('Iter: %d   Loss: %.2f' % (i, loss.data))
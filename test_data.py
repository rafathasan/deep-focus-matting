#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.data import Data

# h = Data()
# h.load()

d = Data()
df = d.generate_dataframe()

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.2)

print(test)
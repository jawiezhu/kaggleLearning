# -*- coding:utf-8 -*-

import pandas as pd
from MultiColumnLabelEncoder import *

fruit_data = pd.DataFrame({
    'fruit':  ['apple','orange','pear','orange'],
    'color':  ['red','orange','green','green'],
    'weight': [5,6,3,4]
})


MultiColumnLabelEncoder(columns = ['fruit','color']).fit_transform(fruit_data)
print(fruit_data)


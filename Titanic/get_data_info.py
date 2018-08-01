# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:23:54 2017

@author: jawiezhu
"""

import pandas as pd


def get_data_report(FILENAME):
    all_df = pd.read_csv(FILENAME)
    REPORT = open(FILENAME.split('.')[0]+'_REPORT', 'a+')
        
    REPORT.write('\nthis file has '.upper() + str(all_df.shape[0]) + ' rows and '.upper() + str(all_df.shape[1]) + ' columns.'.upper())
    REPORT.write('\n')
    
    REPORT.write('\nthe detail of the dataframe:\n'.upper())
    REPORT.write(str(all_df.describe()))

    REPORT.write('\n')
    REPORT.write('\nthe maximum, minimum, mean and median of the numeric col as follows:\n'.upper())
    for col in all_df.columns:
        colType = all_df.dtypes[col]
        if colType == 'int64' or colType == 'float64':
            REPORT.write(col + '\tMAX: ' + str(all_df[col].max()) + '\tMIN: ' + str(all_df[col].min())+ \
                                       '\tMEAN: ' + str(all_df[col].mean())+\
                                                  '\tMEDIAN: ' + str(all_df[col].median()))
            REPORT.write('\n')
    REPORT.write('\n')

    REPORT.write('\nthis file\'s type of the every columns as follows:\n'.upper())
    
    REPORT.write(str(all_df.dtypes))
    REPORT.write('\n')    
    REPORT.write('\nthe top 10 rows in this file:\n'.upper())
    
    REPORT.write(str(all_df.head(10)))
    
    REPORT.write('\n')    
    REPORT.write('\nthe number of the null in every column:\n'.upper())
    REPORT.write( str(pd.isnull(all_df).sum()))
                
    REPORT.close()

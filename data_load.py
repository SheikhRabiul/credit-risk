# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:39:57 2019

@author: Sheikh Rabiul Islam
Purpose: Data load
    load cleaned and extracted data from raw table to original table
"""

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time

conn = sqlite3.connect("database/credit.sqlite")
curr = conn.cursor()

# copy data
curr.execute("insert into origin select * from origin_raw;")
conn.commit()
curr.execute("insert into performance select * from performance_raw;")
conn.commit()

df_result = pd.read_sql_query("select a.*,b.* from performance a, origin b  where a.loanSequenceNumber=b.loanSequenceNumber;", conn)
df_result.to_csv("data/result.csv",encoding='utf-8')

curr.close()
conn.close()

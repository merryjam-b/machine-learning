#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('C:/Users/user/data/data/train_data/train_task_3_4.csv')
data.head()


# In[4]:


for k,df in data.groupby('QuestionId'):
    print(k, len(df), df['IsCorrect'].mean(), df['IsCorrect'].sem())


# In[14]:


df['IsCorrect'].mean()


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/user/data/data/train_data/train_task_3_4.csv')

question_stats = data.groupby('QuestionId')['IsCorrect'].agg(['mean'])

x_values = question_stats.index
y_values = question_stats['mean']

plt.bar(x_values, y_values)
plt.xlabel('Question ID')
plt.ylabel('Mean')
plt.show()


# In[36]:


grouped_data = data.groupby('QuestionId')['IsCorrect'].mean().sort_values(ascending=False)
print(grouped_data)


# In[25]:


eval_validation = pd.read_csv('C:/Users/user/data/data/test_data/quality_response_remapped_public.csv')
print(len(eval_validation))
eval_validation.head()


# In[26]:


eval_validation['score'] = eval_validation.filter(regex='^T', axis = 1).mean(axis=1)
eval_validation['score'].hist()


# In[27]:


import numpy as np
def calc_preference(scores):
    preference = np.ones(len(scores), dtype=int)

    idx_two = scores > 1.5
    preference[idx_two] = 2
    
    return list(preference)

eval_validation['preference'] = calc_preference(eval_validation['score'])
# eval_dev[]
# eval_dev['score']
eval_validation['preference'].hist()


# In[ ]:





# In[ ]:





# 

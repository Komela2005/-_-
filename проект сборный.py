#!/usr/bin/env python
# coding: utf-8

# ## Введение

# Название проекта - Сборный проект №2
# 
# Цель проекта - выяснить, нужно ли менять шрифт в приложении.
# 
# Этапы исследования:
# - подготовка необходимых данных
# - проверка данных
# - изучение воронки данных
# - изучение результатов эксперимента
# 

# ## Подготовка данных

# Загрузим необходимые библиотеки

# In[2]:


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import scipy.stats as st
import math 
import seaborn as sns


# In[3]:


data = pd.read_csv('/datasets/logs_exp.csv', sep='\t')
data


# In[4]:


data.info()


# Количество срок - 244126, столбцов 4 

# Изменим названия столбцов

# In[5]:


data = data.rename(columns={'EventName': 'event_name', 
                            'DeviceIDHash': 'user_id', 
                            'EventTimestamp': 'event_timestamp', 
                            'ExpId': 'exp_id'})


# Посчитаем количество пропусков

# In[5]:


data.isna().sum()


# Посчитаем количество дубликатов

# In[6]:


data.duplicated().sum()


# Удалим из данных дубликаты

# In[7]:


data = data.drop_duplicates()


# Добавим столбцы с датой и временем и отдельно с датой

# In[8]:


data['date_time'] = pd.to_datetime(data['event_timestamp'], unit='s') 
data['date'] = data['date_time'].dt.date
data


# ## Проверка данных

# Убедимся, что ни один пользователь не попал в несколько групп одновременно

# In[9]:


not_uniq_users = data.groupby(['user_id']).agg({'exp_id' : 'nunique'}).reset_index()
print(f'Пользователей, попавших в обе группы: {not_uniq_users[not_uniq_users.exp_id != 1]["user_id"].count()}')


# Посчитаем количество уникальных пользователей

# In[10]:


print('Количество уникальных пользователей:', len(data['user_id'].unique()))


# In[11]:


event_user = data.groupby('user_id')['event_name'].count()
event_user.mean()


# В среднем на каждого пользователя приходится около 32 событий

# Изучим период за которые были получены данные

# In[12]:


print(f'Минимальная дата: {data["date"].min()}')
print(f'Максимальная дата: {data["date"].max()}')


# In[13]:


events_per_date = data['date_time'].value_counts()
events_per_date


# Построим столбчатую диаграмму, которая отобразит количество событий в зависимости от времени в разрезе групп

# In[14]:


series = data.groupby(['date', 'exp_id'])[['event_name']].count().reset_index()

plt.figure(figsize=(15, 7))
ax = sns.barplot(x='date', y='event_name', hue='exp_id', data=series, palette='pastel')

plt.title('Распределение событий по времени')
plt.xlabel('Дата')
plt.ylabel('Количество событий')
plt.legend(title='Группа')
plt.show()


# Судя по графику, полные данные имеются только с 1 по 7 августа, поэтому можно оставить данные только за эти даты

# In[15]:


date = pd.to_datetime('2019-08-01', format = '%Y-%m-%d')
data_new = data[data['date'] >= date]


# In[16]:


data_new.info()


# Теперь количество строк в данных - 240887. Удалено всего 1,32% данных.

# In[17]:


print('Количество уникальных пользователей:', len(data_new['user_id'].unique()))


# In[18]:


data_new['exp_id'].value_counts()


# Количество уникальных пользователей сократилось на 1%

# In[19]:


print('Потеря пользователей в колличественном виде:', len(data['user_id'].unique()) - len(data_new['user_id'].unique()))
print('Потеря пользователей в процентном виде:', (1 - (len(data_new['user_id'].unique()) / len(data['user_id'].unique()))) * 100 )


# ## Изучение воронки данных

# Посчитаем количество событий

# In[20]:


data_new['event_name'].value_counts()


# Количество событий:
# - "MainScreenAppear" (появление главного экрана) - 117328
# - "OffersScreenAppear" (появление предложений на экране) - 46333
# - "CartScreenAppear" (появление экрана корзины) - 42303
# - "PaymentScreenSuccessful" (появление экрана оплаты) - 33918
# - "Tutorial" (обучение) - 1005

# Отсортируем события по числу пользователей

# In[21]:


event_users = data_new.groupby('event_name')['user_id'].nunique().sort_values(ascending=False).to_frame().reset_index()
event_users


# Посчитаем долю пользователей, которые хоть раз совершали событие

# In[22]:


total_users = data_new['user_id'].nunique()
print('Общее количество уникальных пользователей, совершивших события:', total_users)


# In[23]:


event_users['ratio'] = (event_users['user_id'] / total_users).round(3) * 100 
event_users


# Можно предположить, что собятия происходят в таком порядке:
# - MainScreenAppear
# - OffersScreenAppear
# - CartScreenAppear
# - PaymentScreenSuccessful

# Порядок похож на топ по количеству для каждого события, однако этап обучения люди чаще всего пропускают, поэтому это событие можно исключить из воронки

# По воронке событий посчитаем, какая доля пользователей проходит на следующий шаг воронки

# In[24]:


event_users.drop([4], axis=0, inplace=True)
event_users['funnel'] = 1
for i in range(1, 4):
    event_users.loc[i, 'funnel'] = event_users.loc[i, 'user_id'] / (event_users.loc[i-1, 'user_id'])

event_users


# Вывод: 
# - 62% пользователей переходят от этапа "MainScreenAppear" на "OffersScreenAppear"
# - 81% пользователей переходят от этапа "OffersScreenAppear" на "CartScreenAppear"
# - 94% пользователей переходят от этапа "CartScreenAppear" на "PaymentScreenSuccessful"
# 
# Больше всего пользователей теряется на этапе перехода от "MainScreenAppear" до "OffersScreenAppear" -около 28%.
# 
# От первого события по покупки доходит около 47% пользователей

# **Выводы:**
# 
# Количество событий:
# 
# - "MainScreenAppear" (появление главного экрана) - 117328
# - "OffersScreenAppear" (появление предложений на экране) - 46333
# - "CartScreenAppear" (появление экрана корзины) - 42303
# - "PaymentScreenSuccessful" (появление экрана оплаты) - 33918
# - "Tutorial" (обучение) - 1005
# 
# События по числу пользователей:
# 
# - "MainScreenAppear" - 7419
# - "OffersScreenAppear" - 4593
# - "CartScreenAppear" - 3734
# - "PaymentScreenSuccessful" - 3539
# 
# События на сайте происходят в таком порядке:
# 
# - MainScreenAppear
# - Tutorial
# - OffersScreenAppear
# - CartScreenAppear
# - PaymentScreenSuccessful
# 
# 
# - 62% пользователей переходят от этапа "MainScreenAppear" на "OffersScreenAppear".
# - 81% пользователей переходят от этапа "OffersScreenAppear" на "CartScreenAppear".
# - 94% пользователей переходят от этапа "CartScreenAppear" на "PaymentScreenSuccessful".
# 
# Больше всего пользователей теряется на этапе перехода от "MainScreenAppear" до "OffersScreenAppear" -около 28%.
# 
# От первого события по покупки доходит около 47% пользователей

# ## Изучения результатов эксперимента

# Посчитаем количество пользователей в каждой группе

# In[25]:


data_new.groupby('exp_id')['user_id'].nunique().to_frame()


# Количество пользователей в каждой группе примерно одинаково

# Есть 2 контрольные группы для А/А-эксперимента, чтобы проверить корректность всех механизмов и расчётов. Проверьте, находят ли статистические критерии разницу между выборками 246 и 247.

# In[26]:


group_246 = data_new.query('exp_id == 246').groupby('exp_id')['user_id'].nunique().reset_index()
group_247 = data_new.query('exp_id == 247').groupby('exp_id')['user_id'].nunique().reset_index()
ratio = 1 - group_246.iloc[0]['user_id'] / group_247.iloc[0]['user_id']
print('Разница между выборками 246 и 247 составляет {:.2%}'.format(ratio))


# Выберите самое популярное событие. Посчитайте число пользователей, совершивших это событие в каждой из контрольных групп. Посчитайте долю пользователей, совершивших это событие. Проверьте, будет ли отличие между группами статистически достоверным. Проделайте то же самое для всех других событий (удобно обернуть проверку в отдельную функцию). Можно ли сказать, что разбиение на группы работает корректно?

# Посмотрим, сколько человек из каждой группы задействованы в каждом событии

# In[27]:


data2 = data_new.pivot_table(
    index='exp_id', 
    columns='event_name', 
    values='user_id', 
    aggfunc='nunique')
data2 = data2.reset_index()
data2


# События распределены равномерно по каждой группе

# Сформулируем гипотезы:
# 
# - Н0: доли пользователей на каждом этапе в группах одинаковы
# - Н1: доли пользователей на каждом этапе в группах различны

# Данные гипотезы будут проверяться Z-тестом, т.к. размер выборки достаточно велик. Уровень статистической значимости - 0.05

# In[28]:


def z_test(dataframe, num_group, alpha):
    #пользователи в каждой группеm
    users = [dataframe.query('exp_id == @group')['user_id'].nunique() for group in num_group]
    #перебор экспериментальных групп по событиям и количество пользователей
    for event in dataframe.event_name.unique():
        events = [dataframe.query('exp_id == %d and event_name == "%s"' % (group, event))['user_id'].nunique() for group in num_group]
        
        p1 = events[0] / users[0] 
        p2 = events[1] / users[1] 
        
        p_combined = sum(events) / sum(users) 
        
        difference = p1 - p2 
        
        z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1 / users[0] + 1 / users[1]))
        
        distr = st.norm(0, 1) 
        p_value = (1 - distr.cdf(abs(z_value))) * 2
        
        print('Событие:', event)
        print('p_value: {p_value:}'.format(p_value=p_value))
        
        if p_value < alpha:
            print('Отвергаем нулевую гипотезу')
        else:
            print('Не получилось отвергнуть нулевую гипотезу')
            
        print('')    


# In[29]:


z_test(data_new, [246, 247], 0.05)


# А/А тест, проводимый чтобы убедиться, что контрольные выборки были сформированы верно, показал, что статистически значимых отличий между двумя контрольными выборками нет, следовательно, они были сформированы верно и можно проводить А/В тест.
# 

# Проведем такой же тест для группы контрольной и экспериментальной (246)

# In[30]:


z_test(data_new, [246, 248], 0.05)


# Различия между группами отсутствуют

# Проведем тест для контрольной группы и экспериментальной (247)

# In[31]:


z_test(data_new, [247, 248], 0.05)


# Объединим две контрольные выборки с экспериментальной

# In[32]:


data3 = data_new.copy()
data3['exp_id'].replace({247: 246}, inplace=True)
z_test(data3, [246, 248], 0.05)


# Различий между объединенной группой и экспериментальной снова нет, значит эксперимент можно считать неудачным.

# При проверке гипотез был выбран уровень значимости alpha = 0.05. Всего было выполнено 20 тестов

# ## Выводы

# На этапе подготовки данных были выполнены следущие шаги:
# - загружены необходимые библотеки
# - посчитано количество пропусков и удалены все дубликаты
# - добавлены столбцы с датой и временем и отдельно с датой
# 
# На этапе проверки данных были выполнены следующие шаги:
# - проведена проверка, чтобы ни один пользователь не попал в обе группы
# - посчитано количество уникальных пользователей - 7551
# - среднее количество событий на каждого пользователя - 32
# - минимальная дата полученных данных - 25 июля, максимальная - 7 августа, было решено оставить данные за период с 1 по 7 августа
# 
# На этапе изучения воронки были выполнены следующие шаги:
# - подчситано количество событий
# - подсчитана  доля пользователей, которые хоть раз совершали кжадое событие
# 
# Больше всего пользователей теряется на этапе перехода от "MainScreenAppear" до "OffersScreenAppear" -около 28%.
# 
# От первого события по покупки доходит около 47% пользователей
# 
# Сформулированы гипотезы:
# 
# - Н0: доли пользователей на каждом этапе в группах одинаковы
# - Н1: доли пользователей на каждом этапе в группах различны
# 
# При проверке гипотез был выбран уровень значимости alpha = 0.05. Всего было выполнено 20 тестов.
# По результатам теста выяснилось, что пользователи были распределены по группам равномерно и различий обнаружено не было.
# Различий между объединенной из двух контрольных групп и экспериментальной также нет, следовательно, ухудшение поведения не обнаружено, поэтому можно поменять шрифт в приложении
# 

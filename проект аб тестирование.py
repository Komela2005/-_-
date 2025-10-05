#!/usr/bin/env python
# coding: utf-8

# ## Описание данных

# Импортируем все необходимые библиотеки

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import scipy.stats as stats


# In[2]:


part1 = pd.read_csv('/datasets/hypothesis.csv')
part1


# In[3]:


pd.set_option('display.max_colwidth', None) 


# In[4]:


part2 = pd.read_csv('/datasets/orders.csv')
part2


# In[5]:


part2.info()


# Приведем дату в нужный формат

# In[6]:


part2['date'] = part2['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))


# In[7]:


data = pd.read_csv('/datasets/visitors.csv')
data


# In[8]:


data['date'] = data['date'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))


# In[9]:


data.info()


# Проверим данные на наличие пропусков

# In[10]:


print(part1.isna().sum())
print(part2.isna().sum())
print(data.isna().sum())


# Проверим данные на наличие дубликатов

# In[11]:


print(part1.duplicated().sum())
print(part2.duplicated().sum())
print(data.duplicated().sum())


# Для удобства выведем названия каждой гипотезы полностью

# Проверим количество групп в тесте

# In[12]:


part2['group'].unique()


# Выявим период проведения теста

# In[13]:


print(part2['date'].min())
print(part2['date'].max())


# Дата начала теста - 1 августа 2019, дата окончания теста - 31 августа 2019 года

# Посчитаем количество пользователей в каждой группе

# In[14]:


part2[part2['group'] == "A"].count()


# In[15]:


part2[part2['group'] == "B"].count()


# В группе В оказалось на 83 пользователя больше, однако удалять их не стоит, т.к. это слишком большое число, поэтому оставим так

# Найдем пользователей, которые встречаются в обеих групппах

# In[16]:


users_in_both_groups = part2.groupby("visitorId")["group"].nunique()
users_in_both_groups


# Проверим данные на наличие пользователей, попавших в обе группы

# In[17]:


duplicated_users = users_in_both_groups[users_in_both_groups > 1].index
duplicated_users_count = users_in_both_groups[users_in_both_groups > 1].count()
duplicated_users_count


# Число пользователей, попавших в обе группы - 58. Логично будет удалить их из данных, чтобв избежать искажения

# In[18]:


part2 = part2[~part2["visitorId"].isin(duplicated_users)]
part2["visitorId"].count()


# In[19]:


part1['Hypothesis']


# In[20]:


pd.reset_option('display.max_colwidth')


# ## Часть 1. Приоритизация гипотез.

# Рассчитаем наиболее перспективные гипотезы по ICE

# In[21]:


part1['ICE'] = part1['Impact'] * part1['Confidence'] / part1['Efforts']
print(part1[['Hypothesis', 'ICE']].sort_values(by='ICE', ascending=False))


# **Три наиболее перспективные гипотезы по ICE:**
# - Запустить акцию, дающую скидку на товар в день рождения
# - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей
# - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок

# Рассчитаем наиболее перспективные гипотезы по RICE

# In[22]:


part1['RICE'] = (part1['Reach'] * part1['Impact'] * part1['Confidence']) / part1['Efforts']
print(part1)
print(part1[['Hypothesis', 'RICE']].sort_values(by='RICE', ascending=False))


# **Три наиболее перспективные гипотезы по RICE:**
# - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок
# - Добавить блоки рекомендаций товаров на сайт интернет магазина, чтобы повысить конверсию и средний чек заказа
# - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей

# Причины различий в топах гипотез:
# - гипотеза 8 вошла в топ ICE, но не вошла в RICE, потому что у нее слабое значение параметра Reach, который не учитывается в ICE. Это означает, что пользователи неохотно будут пользоваться акцией, дающей скидку в день рождения
# - гипотеза 2 вошла в RICE, но не вошла в ICE, потому что у нее слабое значение параметра Efforts. Это означает, что гипотезу довольно легко можно проверить

# ## Часть 2. Анализ A/B-теста

# Постройте график кумулятивной выручки по группам. Сделайте выводы и предположения.

# Создадим датафрейм datesGroups с уникальными парами значений 'date' и 'group', таблицы orders. Избавимся от дубликатов методом drop_duplicates()

# In[23]:


datesGroups = part2[['date', 'group']].drop_duplicates()


# Объявим переменную ordersAggregated, содержащую:
# - дату;
# - группу A/B-теста;
# - число уникальных заказов в группе теста по указанную дату включительно;
# - число уникальных пользователей, совершивших хотя бы 1 заказ в группе теста по указанную дату включительно;
# - суммарную выручку заказов в группе теста по указанную дату включительно.

# In[24]:


ordersAggregated = datesGroups.apply(lambda x: part2[np.logical_and(part2['date'] <= x['date'], part2['group'] == x['group'])].agg({'date' : 'max', 'group' : 'max', 'transactionId' : 'nunique', 'visitorId' : 'nunique', 'revenue' : 'sum'}), axis=1).sort_values(by=['date','group'])
ordersAggregated


# Объявим переменную visitorsAggregated, содержащую:
# - дату;
# - группу A/B-теста;
# - количество уникальных посетителей в группе теста по указанную дату включительно.

# In[25]:


visitorsAggregated = datesGroups.apply(lambda x: data[np.logical_and(data['date'] <= x['date'], data['group'] == x['group'])].agg({'date' : 'max', 'group' : 'max', 'visitors' : 'sum'}), axis=1).sort_values(by=['date','group'])
visitorsAggregated


# Определим переменную cumulativeData, объединив ordersAggregated и visitorsAggregated

# In[26]:


cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']
cumulativeData


# Объявим переменные cumulativeRevenueA и cumulativeRevenueB, в которых сохраним данные о датах, выручке и числе заказов в группах A и B.

# In[27]:


cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]
cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]


# **Построим график кумулятивной выручки по группам**

# In[28]:


plt.figure(figsize=(10, 5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B')
plt.title('График кумулятивной выручки по группам')
plt.legend() ;


# Как видно из графика, в самом начале теста выручка была примерно одинакова, после чего выручка группы В стабильно выше группы А.
# На графике также заметен резкий скачок выручки у группы В. С одной стороны это может означать, что группа В действительно стабильно приносит больше выручки, поэтому изменения в группе В довольно успешны. С другой стороны, это может означать наличие выбросов в группе В, поэтому нужно проверить средние чеки в обеих группах

# **Построим график кумулятивного среднего чека по дням**

# In[29]:


plt.figure(figsize=(10, 5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B')
plt.legend()
plt.title('График кумулятивного среднего чека по дням');


# Как видно, в середине теста в группе В произошел резкий скачок в среднем чеке, после чего он был стабильно гораздо больше, чем в группе А

# **Построим график относительно изменения кумулятивного среднего чека группы B к группе A**

# In[30]:


mergedCumulativeRevenue = cumulativeRevenueA.merge(cumulativeRevenueB, left_on='date', right_on='date', how='left', suffixes=['A', 'B'])
plt.figure(figsize=(10, 5))
plt.plot(mergedCumulativeRevenue['date'], (mergedCumulativeRevenue['revenueB']/mergedCumulativeRevenue['ordersB'])/(mergedCumulativeRevenue['revenueA']/mergedCumulativeRevenue['ordersA'])-1)
plt.axhline(y=0, color='black', linestyle='--') 
plt.title('График относительно изменения кумулятивного среднего чека группы B к группе A');


# Из графика видно, что результаты резко менялись несколько раз - рост в начале, резкое падение к середине, затем снова рост. В эти даты были совершены аномальные заказы

# Постройте график кумулятивного среднего количества заказов на посетителя по группам. Сделайте выводы и предположения.

# Добавим в cumulativeData столбец 'conversion' c отношением числа заказов к количеству пользователей в указанной группе в указанный день.

# In[31]:


cumulativeData['conversion'] = cumulativeData['orders']/cumulativeData['visitors']


# Объявим переменные cumulativeDataA и cumulativeDataB, в которых сохраним данные о заказах в сегментах A и B соответственно.

# In[32]:


cumulativeDataA = cumulativeData[cumulativeData['group']=='A']
cumulativeDataB = cumulativeData[cumulativeData['group']=='B']


# **Построим графики кумулятивного среднего количества заказов на посетителя по группам и по дням.**

# In[33]:


plt.figure(figsize=(10, 5))
plt.plot(cumulativeDataA['date'], cumulativeDataA['conversion'], label='A')
plt.plot(cumulativeDataB['date'], cumulativeDataB['conversion'], label='B')
plt.legend()
plt.title('График кумулятивного среднего количества заказов на посетителя по группам и по дням');


# Как видно из графика,в самом начале теста кумулятивное значение среднего числа заказов было больше у группы А, однако потом группа В была стабильно выше

# **Построим график относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A**
# 

# In[34]:


mergedCumulativeConversions = cumulativeDataA[['date','conversion']].merge(cumulativeDataB[['date','conversion']], left_on='date', right_on='date', how='left', suffixes=['A', 'B'])
plt.figure(figsize=(10, 5))
plt.plot(mergedCumulativeConversions['date'], mergedCumulativeConversions['conversionB']/mergedCumulativeConversions['conversionA']-1)
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=0.2, color='grey', linestyle='--')
plt.title('График относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A');


# На протяжении практически всего теста у группы В наблюдается прирост по сравнению с группой А в минимум 10%

# Подсчитаем количество заказов по пользователям
# 

# In[35]:


ordersByUsers = (
    part2.groupby('visitorId', as_index=False)
    .agg({'transactionId': 'nunique'})
)

ordersByUsers.columns = ['userId', 'orders']
ordersByUsers.sort_values(by='orders', ascending=False).head(10)


# **Построим точечную диаграмму числа заказов на одного пользователя:**
# 
# 

# In[36]:


x_values = pd.Series(range(0,len(ordersByUsers)))

plt.scatter(x_values, ordersByUsers['orders']) 
plt.title('Точечная диаграмма числа заказов на одного пользователя');


# Судя по диаграмме, больше всего пользователей делает не более 1 заказа
# 

# **Посчитаем 95-й и 99-й перцентили количества заказов на пользователя**

# In[37]:


print(np.percentile(ordersByUsers['orders'], [95, 99]))


# Не более 5 процентов людей совершают заказы более 2 раз, не более 1 процента - 2 раза. Граница для определения аномального числа заказов - 2 заказа.

# **Построим точечную диаграмму стоимости заказа на одного пользователя:**

# In[38]:


x_values = pd.Series(range(0,len(part2['revenue'])))
plt.scatter(x_values, part2['revenue']) 
plt.title('Точечная диаграмма стоимости заказа на одного пользователя');


# По графику можно заметить, что большинство заказов имели стоимость до 100 тысяч рублей. Есть два выброса - около 1.2 млн рублей и 200 тыс рублей

# **Посчитаем 95-й и 99-й перцентили стоимости заказов**

# In[39]:


print(np.percentile(part2['revenue'], [95, 99]))


# Не более 5% людей совершили покупки на сумму более 26785 и не более 1% совершили покупку на сумму 53904. Аномальными значениями можно считать сумму, превышающую значение 99 перцентиля

# **Посчитаем статистическую значимость различий в среднем количестве заказов на посетителя между группами по «сырым» данным.
# Для начала подготовим данные**

# In[40]:



visitorsADaily = data[data['group'] == 'A'][['date', 'visitors']]
visitorsADaily.columns = ['date', 'visitorsPerDateA']

visitorsACummulative = visitorsADaily.apply(
    lambda x: visitorsADaily[visitorsADaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateA': 'sum'}
    ),
    axis=1,
)
visitorsACummulative.columns = ['date', 'visitorsCummulativeA']

visitorsBDaily = data[data['group'] == 'B'][['date', 'visitors']]
visitorsBDaily.columns = ['date', 'visitorsPerDateB']

visitorsBCummulative = visitorsBDaily.apply(
    lambda x: visitorsBDaily[visitorsBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'visitorsPerDateB': 'sum'}
    ),
    axis=1,
)
visitorsBCummulative.columns = ['date', 'visitorsCummulativeB']


# In[41]:


ordersADaily = (
    part2[part2['group'] == 'A'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersADaily.columns = ['date', 'ordersPerDateA', 'revenuePerDateA']

ordersACummulative = ordersADaily.apply(
    lambda x: ordersADaily[ordersADaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateA': 'sum', 'revenuePerDateA': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersACummulative.columns = [
    'date',
    'ordersCummulativeA',
    'revenueCummulativeA',
]

ordersBDaily = (
    part2[part2['group'] == 'B'][['date', 'transactionId', 'visitorId', 'revenue']]
    .groupby('date', as_index=False)
    .agg({'transactionId': pd.Series.nunique, 'revenue': 'sum'})
)
ordersBDaily.columns = ['date', 'ordersPerDateB', 'revenuePerDateB']

ordersBCummulative = ordersBDaily.apply(
    lambda x: ordersBDaily[ordersBDaily['date'] <= x['date']].agg(
        {'date': 'max', 'ordersPerDateB': 'sum', 'revenuePerDateB': 'sum'}
    ),
    axis=1,
).sort_values(by=['date'])
ordersBCummulative.columns = [
    'date',
    'ordersCummulativeB',
    'revenueCummulativeB',
]

data2 = (
    ordersADaily.merge(
        ordersBDaily, left_on='date', right_on='date', how='left'
    )
    .merge(ordersACummulative, left_on='date', right_on='date', how='left')
    .merge(ordersBCummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsADaily, left_on='date', right_on='date', how='left')
    .merge(visitorsBDaily, left_on='date', right_on='date', how='left')
    .merge(visitorsACummulative, left_on='date', right_on='date', how='left')
    .merge(visitorsBCummulative, left_on='date', right_on='date', how='left')
)

print(data2.head)


# In[42]:


ordersByUsersA = (
    part2[part2['group'] == 'A']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersA.columns = ['userId', 'orders']

ordersByUsersB = (
    part2[part2['group'] == 'B']
    .groupby('visitorId', as_index=False)
    .agg({'transactionId': pd.Series.nunique})
)
ordersByUsersB.columns = ['userId', 'orders'] 


# In[43]:


sampleA = pd.concat([ordersByUsersA['orders'],pd.Series(0, index=np.arange(data2['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])), name='orders')],axis=0)

sampleB = pd.concat([ordersByUsersB['orders'],pd.Series(0, index=np.arange(data2['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])), name='orders')],axis=0)


#  Сформулируем гипотезы:
#  - Нулевая: различий в среднем количестве заказов между группами нет. 
#  - Альтернативная: различия в среднем между группами есть.

# In[44]:


display(f"P-value: {stats.mannwhitneyu(sampleA, sampleB)[1]:.3f}")
display(f"Относительный прирост среднего группы В к конверсии группы А: {(sampleB.mean() / sampleA.mean() - 1):.3f}")


# Вывод: p-value 0.017 меньше, чем 0.05. Следовательно, нулевую гипотезу о том, что нет статистически значимых различий в среднем количестве заказов между группами, отвергаем. Выигрыш группы В составляет 13,8%

# **Посчитаем статистическую значимость различий в среднем чеке заказа между группами по «сырым» данным**

#  Сформулируем гипотезы:
#  - Нулевая: различий в среднем чеке заказа между группами нет. 
#  - Альтернативная: различия в среднем чеке заказа между группами есть.

# In[45]:


display(f"P-value: {stats.mannwhitneyu(part2[part2['group']=='A']['revenue'], part2[part2['group']=='B']['revenue'])[1]:.3f}")
display(f"Относительный прирост среднего чека группы В к группе А: {part2[part2['group']=='B']['revenue'].mean()/part2[part2['group']=='A']['revenue'].mean()-1:.3f}")


# Вывод: p-value больше 0.05, следовательно статистически значимых отличий в среднем чеке между группами нет. Однако относительное различие между чеками в группах - 28,7 процентов

# In[46]:



usersWithManyOrders = pd.concat(
    [
        ordersByUsersA[ordersByUsersA['orders'] > 2]['userId'],
        ordersByUsersB[ordersByUsersB['orders'] > 2]['userId'],
    ],
    axis=0,
)
usersWithExpensiveOrders = part2[part2['revenue'] > 26785]['visitorId']
abnormalUsers = (
    pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis=0)
    .drop_duplicates()
    .sort_values()
)
print(abnormalUsers.head(5))
print(abnormalUsers.shape[0]) 
print(part2['visitorId'].count()) #посчитаем общее число пользователей в тесте


# Число аномальных пользователей - 58. Число пользователей в тесте - 1016. Процент аномалий от общего числа пользователей составляет 5.71, что позволяет удалить аномальных пользователей от общего числа

# In[47]:


sampleAFiltered = pd.concat(
    [
        ordersByUsersA[
            np.logical_not(ordersByUsersA['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data2['visitorsPerDateA'].sum() - len(ordersByUsersA['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)

sampleBFiltered = pd.concat(
    [
        ordersByUsersB[
            np.logical_not(ordersByUsersB['userId'].isin(abnormalUsers))
        ]['orders'],
        pd.Series(
            0,
            index=np.arange(
                data2['visitorsPerDateB'].sum() - len(ordersByUsersB['orders'])
            ),
            name='orders',
        ),
    ],
    axis=0,
)


# In[48]:


display(f"P-value: {stats.mannwhitneyu(sampleAFiltered, sampleBFiltered)[1]:.3f}")
display(f"Относительный прирост среднего чека группы В к группе А: {sampleBFiltered.mean()/sampleAFiltered.mean()-1:.3f}")


# Результаты теста практически не изменились

# In[49]:


display(f"P-value: {stats.mannwhitneyu(part2[np.logical_and(part2['group'] == 'A',np.logical_not(part2['visitorId'].isin(abnormalUsers)),)]['revenue'],part2[np.logical_and(part2['group'] == 'B',np.logical_not(part2['visitorId'].isin(abnormalUsers)),)]['revenue'],)[1]:.3f}")
display(f"Изменения чека группы В относительно группы А: {part2[np.logical_and(part2['group'] == 'B',np.logical_not(part2['visitorId'].isin(abnormalUsers)),)]['revenue'].mean()/ part2[np.logical_and(part2['group'] == 'A',np.logical_not(part2['visitorId'].isin(abnormalUsers)),)]['revenue'].mean()- 1:.3f}")


# Статистически значимых отличий в среднем чеке между группами снова нет, однако теперь изменения в чеке составляют 4.8%, т.е. чек группы В стал на 4.8% меньше, первый же тест показал что средний чек группы В на 18.2% больше. Можно сделать вывод, что у группы В было много аномальных чеков, из-за чего результаты теста ранее показывали другое значение.

# Удалим аномальные значения из данных

# In[50]:


part2_good = part2.query('not visitorId in @abnormalUsers')


# 
# Построим график относительного изменения среднего чека на посетителя группы B к группе A

# In[51]:


new_data = part2_good.pivot_table(index='date', columns='group', 
                                        aggfunc={'transactionId': 'nunique','revenue': 'sum'}).sort_index()
pivot_data = new_data.cumsum().reset_index()
pivot_data['new_revenue'] = (
    (pivot_data[('revenue', 'B')]  / pivot_data[('transactionId', 'B')] ) 
    / (pivot_data[('revenue', 'A')]  / pivot_data[('transactionId', 'A')] ) - 1
)


# In[52]:


pivot_data.plot('date', 'new_revenue', figsize=(10, 5))
plt.axhline(y=0, color='black', linestyle='--') 
plt.title('График относительного изменения среднего чека на посетителя группы B к группе A')
plt.show();


# Как видно из графика, поведение чека нестабильно, однако средний чек группы В большинство времени имеет меньшее значение, чем у группы А

# ## Выводы

# **В первой части проекта "Описание данных" были выполнены следующие шаги:**
# 
# - импортированы необходимые библиотеки
# - данные были проверены на наличие пропусков и дубликатов
# - изменены типы даты
# 
# Была проведена базовая проверка данных:
# - дата начала теста - 1 августа 2019 года
# - дата окончания теста - 31 августа 2019 года
# - количество участников - 1197. Количество участников группы А - 557, группы В - 640.
# - в данных встретились участники, попавшие в обе группы одновременно, было принято решение удалить их из общего числа данных
# - метрика, которую необходимо улучшить в процессе теста - величина среднего чека на пользователя
# **Во второй части проекта "Приоритезация гипотез" были рассчитаны три самые приоритетные гипотеза по ICE:**
# 
# - Запустить акцию, дающую скидку на товар в день рождения
# - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей
# - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок
# 
# Также были рассчитаны наиболее перспективные гипотезы по RICE:
# 
# - Добавить форму подписки на все основные страницы, чтобы собрать базу клиентов для email-рассылок
# - Добавить блоки рекомендаций товаров на сайт интернет магазина, чтобы повысить конверсию и средний чек заказа
# - Добавить два новых канала привлечения трафика, что позволит привлекать на 30% больше пользователей
# 
# Причинами различий в топах можно считать низкие значения параметров Reach и Efforts в некоторых гипотезах
# 
# **В третьей части проекта "Анализ А/Б теста" были выполнены следующие шаги:**
# 
# - Построен график кумулятивной выручки по группам, группа В была стабильно выше А
# - Построен график кумулятивного среднего чека по дням, группа В также лидировала почти все время
# - Построен графики кумулятивного среднего количества заказов на посетителя по группам и по дням. Группа В была стабильно выше группы А
# - Построен график относительного изменения кумулятивного среднего количества заказов на посетителя группы B к группе A, группа В также выше.
# - Построена точечная диаграмма числа заказов на одного пользователя. Не более 5 процентов людей совершают заказы более 1 раз, не более 1 процента - 2 раза. Граница для определения аномального числа заказов - 2 заказа.
# - Построена точечная диаграмма стоимости заказа на одного пользователя. Не более 5% людей совершили покупки на сумму более 26785 и не более 1% совершили покупку на сумму 53904. Аномальными значениями можно считать сумму, превышающую значение 95 перцентиля
# 
# **Сформулированы гипотезы:**
# 
# Нулевая: различий в среднем количестве заказов между группами нет.
# 
# Альтернативная: различия в среднем между группами есть.
# 
# По результатам теста Манна-Уитни значение p-value меньше, чем 0.05. Следовательно, нулевую гипотезу о том, что нет статистически значимых различий в среднем количестве заказов между группами, отвергаем. Выигрыш группы В составляет 16%
# 
# **Сформулированы гипотезы:**
# 
# Нулевая: различий в среднем чеке заказа между группами нет.
# 
# Альтернативная: различия в среднем чеке заказа между группами есть.
# 
# По результатам теста Манна-Уитни значение p-value больше 0.05, следовательно статистически значимых отличий в среднем чеке между группами нет. Однако относительное различие между чеками в группах - почти 28.7 процентов
# 
# Тесты были проведены повторно после удаления аномальных значений. Результаты повторного теста показали, что группа В содержала в себе много аномальных значений, из-за чего показатель среднего чека снизился.
# Новый график показал, что значение среднего чека группы В большиинство времени было ниже, чем у группы А. Тест показал, что статистически значимых различий между чеками группы А и В нет.Т.к. величина среднего чека - основная метрика для теста,было принято решение остановить тест и признать его неуспешным

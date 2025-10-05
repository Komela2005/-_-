#!/usr/bin/env python
# coding: utf-8

# ### Загрузите данные и подготовьте их к анализу

# Загрузите данные о визитах, заказах и рекламных расходах из CSV-файлов в переменные.
# 
# **Пути к файлам**
# 
# - визиты: `/datasets/visits_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/visits_info_short.csv);
# - заказы: `/datasets/orders_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/orders_info_short.csv);
# - расходы: `/datasets/costs_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/costs_info_short.csv).
# 
# Изучите данные и выполните предобработку. Есть ли в данных пропуски и дубликаты? Убедитесь, что типы данных во всех колонках соответствуют сохранённым в них значениям. Обратите внимание на столбцы с датой и временем.

# Импортируем необходимые библиотеки и загрузим данные

# In[223]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[224]:


visits = pd.read_csv('/datasets/visits_info_short.csv')
orders = pd.read_csv('/datasets/orders_info_short.csv')
costs = pd.read_csv('/datasets/costs_info_short.csv')


# In[225]:


print(visits)
print(orders)
print(costs)


# Приведем названия столбцов к стандартному виду

# In[226]:


visits.columns = map(str.lower, visits.columns)
orders.columns = map(str.lower, orders.columns)
costs.columns = map(str.lower, costs.columns)


# In[227]:


visits = visits.rename(columns={'user id':'user_id', 'session start': 'session_st', 'session end': 'session_end' })
orders = orders.rename(columns={'user id': 'user_id', 'event dt': 'event_dt'})


# Проверим данные на наличие пропусков

# In[228]:


visits.isna().sum()


# In[229]:


orders.isna().sum()


# In[230]:


costs.isna().sum()


# Пропуски отсутствуют

# Проверим данные на наличие дубликатов

# In[231]:


visits.duplicated().sum()


# In[232]:


orders.duplicated().sum()


# In[233]:


costs.duplicated().sum()


# Дубликаты отсутствуют

# Проверим типы данных

# In[234]:


visits.info()


# В столбцах с датой и временем неверный тип, изменим его

# In[235]:


visits['session_st'] = pd.to_datetime(visits['session_st'])
visits['session_end'] = pd.to_datetime(visits['session_end'])
visits.info()


# In[236]:


orders.info()


# Изменим тип данных в столбце event_dt

# In[237]:


orders['event_dt'] = pd.to_datetime(orders['event_dt'])
orders['event_dt'] 


# На данном этапе были выполнены следующие шаги:
# - загружены необходимые данные
# - импортированы необходимые беблиотеки
# - названия столбцов приведены к стандартному виду
# - данные проверены на наличие пропусков и дубликатов
# - изменены типы данных в столбцах некоторых таблиц
# 

# ### Задайте функции для расчёта и анализа LTV, ROI, удержания и конверсии.
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.
# 
# Это функции для вычисления значений метрик:
# 
# - `get_profiles()` — для создания профилей пользователей,
# - `get_retention()` — для подсчёта Retention Rate,
# - `get_conversion()` — для подсчёта конверсии,
# - `get_ltv()` — для подсчёта LTV.
# 
# А также функции для построения графиков:
# 
# - `filter_data()` — для сглаживания данных,
# - `plot_retention()` — для построения графика Retention Rate,
# - `plot_conversion()` — для построения графика конверсии,
# - `plot_ltv_roi` — для визуализации LTV и ROI.

# Вызовем функцию get_profiles(), чтобы составить профили пользователей по данным сессий из датафрейма

# In[238]:


# добавляем параметр ad_costs — траты на рекламу
def get_profiles(sessions, orders, ad_costs):

    # сортируем сессии по ID пользователя и дате привлечения
    # группируем по ID и находим параметры первых посещений
    profiles = (
        visits.sort_values(by=['user_id', 'session_st'])
        .groupby('user_id')
        .agg(
            {
                'session_st': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first',
            }
        )
         # время первого посещения назовём first_ts
        .rename(columns={'session_st': 'first_ts'})
        .reset_index()  # возвращаем user_id из индекса
    )

    # для когортного анализа определяем дату первого посещения
    # и первый день месяца, в который это посещение произошло
    profiles['dt'] = profiles['first_ts'].dt.date
    profiles['month'] = profiles['first_ts'].astype('datetime64[M]')

    # добавляем признак платящих пользователей
    profiles['payer'] = profiles['user_id'].isin(orders['user_id'].unique())

    # считаем количество уникальных пользователей
    # с одинаковыми источником и датой привлечения
    new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
         # столбец с числом пользователей назовём unique_users
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()  # возвращаем dt и channel из индексов
    )

    # объединяем траты на рекламу и число привлечённых пользователей
    # по дате и каналу привлечения
    ad_costs = ad_costs.merge(new_users, on=['dt', 'channel'], how='left')

    # делим рекламные расходы на число привлечённых пользователей
    # результаты сохраним в столбец acquisition_cost (CAC)
    ad_costs['acquisition_cost'] = ad_costs['costs'] / ad_costs['unique_users']

    # добавим стоимость привлечения в профили
    profiles = profiles.merge(
        ad_costs[['dt', 'channel', 'acquisition_cost']],
        on=['dt', 'channel'],
        how='left',
    )

    # органические пользователи не связаны с данными о рекламе,
    # поэтому в столбце acquisition_cost у них значения NaN
    # заменим их на ноль, ведь стоимость привлечения равна нулю
    profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)
    
    return profiles  # возвращаем профили с CAC


# Вызовем фукнцию для расчёта коэффициента удержания

# In[239]:


def get_retention(
    profiles,
    sessions,
    observation_date,
    horizon_days,
    dimensions=[],  # новый параметр dimensions
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # собираем «сырые» данные для расчёта удержания
    result_raw = result_raw.merge(
        sessions[['user_id', 'session_st']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['session_st'] - result_raw['first_ts']
    ).dt.days

    # рассчитываем удержание
    # новый вариант с dimensions
    result_grouped = result_raw.pivot_table(
        index=dimensions,  # заменили dt
        columns='lifetime',
        values='user_id',
        aggfunc='nunique',
    )
    cohort_sizes = (
        result_raw.groupby(dimensions)  # заменили dt
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'cohort_size'})
    )
    result_grouped = cohort_sizes.merge(
        result_grouped, on=dimensions, how='left'  # заменили dt
    ).fillna(0)
    result_grouped = result_grouped.div(result_grouped['cohort_size'], axis=0)

    # исключаем все лайфтаймы, превышающие горизонт анализа
    result_grouped = result_grouped[
        ['cohort_size'] + list(range(horizon_days))
    ]
    # восстанавливаем столбец с размерами когорт
    result_grouped['cohort_size'] = cohort_sizes

    # возвращаем таблицу удержания и сырые данные
    return result_raw, result_grouped


# Вызовем функцию для расчета конверсии

# In[240]:


def get_conversion(
    profiles,
    orders,
    observation_date,
    analysis_horizon,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=analysis_horizon - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # определяем дату и время первой покупки для каждого пользователя
    first_purchases = (
        orders.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )

    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    # рассчитываем лайфтайм для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users' 
        dimensions = dimensions + ['cohort']

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, analysis_horizon):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        result = result.fillna(0).cumsum(axis=1)
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # делим каждую «ячейку» в строке на размер когорты
        # и получаем conversion rate
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(analysis_horizon))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу конверсии
    result_grouped = group_by_dimensions(result_raw, dimensions, analysis_horizon)

    # для таблицы динамики конверсии убираем 'cohort' из dimensions
    if 'cohort' in dimensions: 
        dimensions = []

    # получаем таблицу динамики конверсии
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], analysis_horizon
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time


# Вызовем функцию для расчета ltv

# In[241]:


def get_ltv(
    profiles,
    orders,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')
    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        orders[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
    )
    # рассчитываем лайфтайм пользователя для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days
    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    # функция группировки по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        # строим «треугольную» таблицу выручки
        result = df.pivot_table(
            index=dims, columns='lifetime', values='revenue', aggfunc='sum'
        )
        # находим сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)
        # вычисляем размеры когорт
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        # объединяем размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # считаем LTV: делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_days))]
        # восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes

        # собираем датафрейм с данными пользователей и значениями CAC, 
        # добавляя параметры из dimensions
        cac = df[['user_id', 'acquisition_cost'] + dims].drop_duplicates()

        # считаем средний CAC по параметрам из dimensions
        cac = (
            cac.groupby(dims)
            .agg({'acquisition_cost': 'mean'})
            .rename(columns={'acquisition_cost': 'cac'})
        )

        # считаем ROI: делим LTV на CAC
        roi = result.div(cac['cac'], axis=0)

        # удаляем строки с бесконечным ROI
        roi = roi[~roi['cohort_size'].isin([np.inf])]

        # восстанавливаем размеры когорт в таблице ROI
        roi['cohort_size'] = cohort_sizes

        # добавляем CAC в таблицу ROI
        roi['cac'] = cac['cac']

        # в финальной таблице оставляем размеры когорт, CAC
        # и ROI в лайфтаймы, не превышающие горизонт анализа
        roi = roi[['cohort_size', 'cac'] + list(range(horizon_days))]

        # возвращаем таблицы LTV и ROI
        return result, roi

    # получаем таблицы LTV и ROI
    result_grouped, roi_grouped = group_by_dimensions(
        result_raw, dimensions, horizon_days
    )

    # для таблиц динамики убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []

    # получаем таблицы динамики LTV и ROI
    result_in_time, roi_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return (
        result_raw,  # сырые данные
        result_grouped,  # таблица LTV
        result_in_time,  # таблица динамики LTV
        roi_grouped,  # таблица ROI
        roi_in_time,  # таблица динамики ROI
    ) 


# Далее вызовем функции для построения графиков

# In[242]:


# функция для сглаживания фрейма

def filter_data(df, window):
    # для каждого столбца применяем скользящее среднее
    for column in df.columns.values:
        df[column] = df[column].rolling(window).mean() 
    return df 


# In[243]:


# функция для визуализации конверсии

def plot_conversion(conversion, conversion_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 5))

    # исключаем размеры когорт
    conversion = conversion.drop(columns=['cohort_size'])
    # в таблице динамики оставляем только нужный лайфтайм
    conversion_history = conversion_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # первый график — кривые конверсии
    ax1 = plt.subplot(1, 2, 1)
    conversion.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия пользователей')

    # второй график — динамика конверсии
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    columns = [
        # столбцами сводной таблицы станут все столбцы индекса, кроме даты
        name for name in conversion_history.index.names if name not in ['dt']
    ]
    filtered_data = conversion_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика конверсии пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show() 


# In[244]:


# функция для визуализации удержания

def plot_retention(retention, retention_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 10))

    # исключаем размеры когорт и удержание первого дня
    retention = retention.drop(columns=['cohort_size', 0])
    # в таблице динамики оставляем только нужный лайфтайм
    retention_history = retention_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # если в индексах таблицы удержания только payer,
    # добавляем второй признак — cohort
    if retention.index.nlevels == 1:
        retention['cohort'] = 'All users'
        retention = retention.reset_index().set_index(['cohort', 'payer'])

    # в таблице графиков — два столбца и две строки, четыре ячейки
    # в первой строим кривые удержания платящих пользователей
    ax1 = plt.subplot(2, 2, 1)
    retention.query('payer == True').droplevel('payer').T.plot(
        grid=True, ax=ax1
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')


# In[245]:


# функция для визуализации LTV и ROI

def plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon, window=7):

    # задаём сетку отрисовки графиков
    plt.figure(figsize=(20, 10))

    # из таблицы ltv исключаем размеры когорт
    ltv = ltv.drop(columns=['cohort_size'])
    # в таблице динамики ltv оставляем только нужный лайфтайм
    ltv_history = ltv_history.drop(columns=['cohort_size'])[[horizon - 1]]

    # стоимость привлечения запишем в отдельный фрейм
    cac_history = roi_history[['cac']]

    # из таблицы roi исключаем размеры когорт и cac
    roi = roi.drop(columns=['cohort_size', 'cac'])
    # в таблице динамики roi оставляем только нужный лайфтайм
    roi_history = roi_history.drop(columns=['cohort_size', 'cac'])[
        [horizon - 1]
    ]

    # первый график — кривые ltv
    ax1 = plt.subplot(2, 3, 1)
    ltv.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('LTV')

    # второй график — динамика ltv
    ax2 = plt.subplot(2, 3, 2, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in ltv_history.index.names if name not in ['dt']]
    filtered_data = ltv_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика LTV пользователей на {}-й день'.format(horizon))

    # третий график — динамика cac
    ax3 = plt.subplot(2, 3, 3, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in cac_history.index.names if name not in ['dt']]
    filtered_data = cac_history.pivot_table(
        index='dt', columns=columns, values='cac', aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика стоимости привлечения пользователей')

    # четвёртый график — кривые roi
    ax4 = plt.subplot(2, 3, 4)
    roi.T.plot(grid=True, ax=ax4)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('ROI')

    # пятый график — динамика roi
    ax5 = plt.subplot(2, 3, 5, sharey=ax4)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in roi_history.index.names if name not in ['dt']]
    filtered_data = roi_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax5)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.xlabel('Дата привлечения')
    plt.title('Динамика ROI пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show() 


# На данном этапе были вызваны функции, необходимые для создания профилей пользователей, расчета retention rate, conversion rate, и ltv, а также были вызваны функции для создания графиков

# <div class="alert alert-success">
# <b>👍 Успех:</b> Нужные функции подготовлены!
# </div>

# ### Исследовательский анализ данных
# 
# - Составьте профили пользователей. Определите минимальную и максимальную даты привлечения пользователей.
# - Выясните, из каких стран пользователи приходят в приложение и на какую страну приходится больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих из каждой страны.
# - Узнайте, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого устройства.
# - Изучите рекламные источники привлечения и определите каналы, из которых пришло больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.
# 
# После каждого пункта сформулируйте выводы.

# In[246]:


costs['dt'] = pd.to_datetime(costs['dt']).dt.date


# In[247]:


profiles = get_profiles(visits, orders, costs)

profiles['acquisition_cost'].sum()


# Составим профили пользователей с помощью функции get_profiles

# In[248]:


profiles = get_profiles(visits, orders, costs)
profiles


# In[249]:


profiles['first_ts'] = pd.to_datetime(profiles['first_ts'])
profiles['first_ts']


# Результат — 150008 пользовательских профилей, в каждом из которых есть данные о дате первого посещения, регионе и рекламном источнике, который мотивировал пользователя посетить приложение.

# Далее определим минимальную и максимальную даты привлечения пользователей

# In[250]:


min_analysis_date = profiles['dt'].min()
observation_date = profiles['dt'].max()# ваш код здесь

print(min_analysis_date, observation_date)


# In[251]:


horizon_days =  14 
max_analysis_date =  observation_date - timedelta(days=horizon_days - 1)
print(max_analysis_date)


# Минимальная дата - 1 мая 2019 года, максимальная - 27 октября 2019 года. С учетом горизонта анализа в 14 дней - максимально возможная дата для анализа - 14 октября 2019
# 

# Сделаем таблицу по странам пользователей

# In[252]:


profiles.groupby('region').agg({'user_id': 'nunique', 'payer': 'mean'}).sort_values(by='user_id', ascending=False)


# Больше всего пользователей из США, меньше всего из Германии. Больше всего платящих пользователей также в США, меньше всего - во Франции

# Сделаем таблицу по устройствам пользователей

# In[253]:


profiles.groupby('device').agg({'user_id': 'nunique', 'payer': 'mean'}).sort_values(by='user_id', ascending=False)


# Клиенты чаще пользуются iPhone (54479), больше всего платящих пользователей также оттуда, реже всего - Мас (30042)

# <div class="alert alert-success">
# <b>👍 Успех:</b> Все верно!
# </div>

# Сделаем таблицу по каналам привлечения пользователей

# In[254]:


profiles.groupby('channel').agg({'user_id': 'nunique', 'payer': 'mean'}).sort_values(by='user_id', ascending=False)


# Самый популярный канал - FaceBoom(56439), наименее популярный - lambdaMediaAds(2149). Больше всего платящих пользователей пришло из канала FaceBoom, меньше - из OppleCreativeMedia

# Выводы
# 
# - Больше всего пользователей из США, меньше всего из Германии. Больше всего платящих пользователей в США, меньше всего - во Франции
# - Клиенты чаще пользуются iPhone (54479), больше всего платящих пользователей также оттуда, реже всего - Мас (30042). Меньше платящих пользователей приходит из РС
# - Самый популярный канал - FaceBoom(56439), наименее популярный - lambdaMediaAds(2149). Больше всего платящих пользователей пришло из канала FaceBoom, меньше - из OppleCreativeMedia

# ### Маркетинг
# 
# - Посчитайте общую сумму расходов на маркетинг.
# - Выясните, как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.
# - Постройте график с визуализацией динамики изменения расходов во времени по неделям по каждому источнику. Затем на другом графике визуализируйте динамику изменения расходов во времени по месяцам по каждому источнику.
# - Узнайте, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. Используйте профили пользователей.
# 
# Напишите промежуточные выводы.

# Посчитаем общую сумму расходов на маркетинг

# In[255]:


costs['costs'].sum()


# Общаа сумма расходов на маркетинг - 105497

# Сделаем таблицу с распределением трат на каждый рекламный источник

# In[256]:


costs.groupby('channel').agg({'costs': 'sum'}).sort_values(by='costs', ascending=False)


# Больше всего денег было потрачено на источник TipTop (54751), меньше всего (если не учитывать organic, т.к. там будет 0) - YRabbit (944)

# Проанализируем САС по каналам привлечения

# In[257]:


costs.pivot_table(
    index='dt', columns='channel', values='costs', aggfunc='sum'
).plot(grid=True, figsize=(10, 5))
plt.ylabel('CAC, $')
plt.xlabel('Дата привлечения')
plt.title('Динамика САС по каналам привлечения')
plt.show()


# У большинства источников САС варьировалась в пределах от 0 до 50 долларов, однако у двух источников (TipTop и FaceBoom) показатель превысил 100 долларов, а в начале октября 2019 САС TipTop достиг 600 долларов

# Проанализируем динамику расходов по неделям

# In[258]:


costs.dt = pd.to_datetime(costs.dt)
costs['week'] = costs.dt.dt.week
costs['month'] = costs.dt.dt.month

plt.figure(figsize=(25, 7))

# задаем недельные расходы
report_week = costs.pivot_table(index='channel', columns='week', values='costs', aggfunc='sum'
)

# Строим динамику расходов по Неделям
report_week.T.plot(
    grid=True, xticks=list(report_week.columns.values), ax=plt.subplot(1, 2, 1)
)
plt.title('Динамика расходов по неделям')


# В динамике расходов по неделям также выдеяются два источника - TipTop и FaceBoom, у обоих источников был пик по расходам на 39 неделе

# Проанализируем расходы по месяцам

# In[259]:


plt.figure(figsize=(25, 7))
# задаем расходы по месяцам
report_month = costs.pivot_table(index='channel', columns='month', values='costs', aggfunc='sum')


# строим изменение расходов по месяцам
report_month.T.plot(
    grid=True, xticks=list(report_month.columns.values), ax=plt.subplot(1, 2, 2)
)
plt.title('Динамика расходов по месяцам')
plt.show()


# Пик расходов по месяцам у двух самых затратных источников: TipTop - сентябрь, FaceBoom - август
# 

# Проанализируем динамику стоимости привлечения одного пользователя 

# In[260]:


# строим график истории изменений CAC по каналам привлечения

profiles.pivot_table(
    index='dt', columns='channel', values='acquisition_cost', aggfunc='mean'
).plot(grid=True, figsize=(10, 5))
plt.ylabel('CAC, $')
plt.xlabel('Дата привлечения')
plt.title('Динамика САС по каналам привлечения')
plt.show()


# Теперь на графике выделяется только один канал привлечения - TipTop, стоимость привлечения одного пользователя превышает 3,5$

# На данном этапе были выполнены следующие шаги:
# 
# - Посчитана общая сумма расходов на маркетинг - 105497
# - Больше всего денег было потрачено на источник TipTop (54751), меньше всего (если не учитывать organic, т.к. там будет 0) - YRabbit (944)
# - У большинства источников САС варьировалась в пределах от 0 до 50 долларов, однако у двух источников (TipTop и FaceBoom) показатель превысил 100 долларов, а в начале октября 2019 САС TipTop достиг 600 долларов
# - В динамике расходов по неделям выдеяются два источника - TipTop и FaceBoom, у обоих источников был пик по расходам на 39 неделе
# - Пик расходов по месяцам у двух самых затратных источников: TipTop - сентябрь, FaceBoom - август
# - Лидер по стоимости привлечения одного пользователя - TipTop (около 3.5 долларов)

# ### Оцените окупаемость рекламы
# 
# Используя графики LTV, ROI и CAC, проанализируйте окупаемость рекламы. Считайте, что на календаре 1 ноября 2019 года, а в бизнес-плане заложено, что пользователи должны окупаться не позднее чем через две недели после привлечения. Необходимость включения в анализ органических пользователей определите самостоятельно.
# 
# - Проанализируйте окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проверьте конверсию пользователей и динамику её изменения. То же самое сделайте с удержанием пользователей. Постройте и изучите графики конверсии и удержания.
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по странам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Ответьте на такие вопросы:
#     - Окупается ли реклама, направленная на привлечение пользователей в целом?
#     - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
#     - Чем могут быть вызваны проблемы окупаемости?
# 
# Напишите вывод, опишите возможные причины обнаруженных проблем и промежуточные рекомендации для рекламного отдела.

# **Исключим из данных пользователей с каналом привлечения organic, чтобы не искажать данные**

# In[261]:


profiles = profiles.query('channel != "organic"')


# In[262]:


# считаем LTV и ROI
ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, horizon_days
)

# строим графики
plot_ltv_roi(ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days) 


# - Из графика САС сразу видно, что стоимость привлечения пользователей резко выросла в июне 2019 года
# - Реклама не окупается, в июне начался резкий спад показателя ROI, что также связано с увеличением стоимости привлечения клиента

# Получим тепловую карту и кривую конверсии

# In[263]:


conversion_raw, conversion, conversion_history = get_conversion(
    profiles, orders, observation_date, horizon_days
)
plt.figure(figsize=(20, 5)) # размер сетки для графиков

# удалите столбец с размерами когорт из таблицы конверсии
report = conversion.drop(columns=['cohort_size'])
# постройте кривые конверсии
report.T.plot(
    grid=True, xticks=list(report.columns.values), ax=plt.subplot(1, 2, 2)
)
plt.title('Кривая конверсии')
# постройте тепловую карту
sns.heatmap(
    report, annot=True, fmt='.2%', ax=plt.subplot(1, 2, 1)
) 
plt.title('Тепловая карта конверсии')

plt.show()


# Посмотрим на удержание для платящих и неплатящих пользователей

# In[264]:


# рассчитываем удержание с учётом совершения покупки

retention_raw, retention = get_retention(
    profiles, visits, observation_date, horizon_days, dimensions=['payer']
)  # передаём параметру dimensions столбец payer

report = retention.drop(columns=['cohort_size', 0])
report.T.plot(grid=True, xticks=list(report.columns.values), figsize=(15, 5))
plt.xlabel('Лайфтайм')
plt.title('Кривые удержания с разбивкой по совершению покупок')
plt.show()


# Удержание платящих пользователей ожидаемо выше, чем у неплатящих

# <div class="alert alert-danger">
#     <s><b>😔 Необходимо исправить:</b> Ячейка не выполняется в тренажере</s>
# </div>

# По графику удержания можно заметить сильные скачки в зависимости от времени

# Разбивка по устройствам

# In[265]:


# смотрим окупаемость с разбивкой по устройствам

dimensions = ['device']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# В графиках с разбивкой по устройствам все довольно однородно, стоимость привлечения выросла для всех, ROI также вырос для всех, однако в особенности нужно отметить неокупаемость рекламы у пользователей Iphone и Mac

# Окупаемость с разбивкой по странам

# In[266]:


# смотрим окупаемость с разбивкой по странам

dimensions = ['region']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# In[267]:


visits.groupby('region').agg({'user_id': 'nunique'}).sort_values(by='user_id', ascending=False)


# Как видно, больше всего пользователей приходит именно из сша

# В разбивке по странам явно выделяется США - только у США вырос резко вырос показатель САС в июне 2019 года, что повлияло на общую статистику, а также только у США реклама не окупается, однако лайфтайм у США самый высокий из остальных стран

# Окупаемость с разбивкой по каналам

# In[268]:


# смотрим окупаемость с разбивкой по источникам привлечения

dimensions = ['channel']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# Исключим источник organic

# In[272]:


visits = visits.query('channel != "organic"')


# In[273]:


visits.groupby('channel').agg({'user_id': 'nunique'}).sort_values(by='user_id', ascending=False)


# Несмотря на отсутствие окупаемости, больше всего людей пришло именно из источников TipTop и FaceBoom, судя по всему, эти источники обладают наименьшим процентом удержания.

# - В разбивке по каналам привлечения выделяется источник TipTop, стоимость привлечения пользователей в нем также резко выросла в ине 2019 года
# - Помимо него, реклама также не окупается у источников FaceBoom и AdNonSense, однако исходя из таблилцы с тратами на рекламные источники, именно на эти каналы привлечения было потрачено больше всего средств

# Проверим удержание для пользователей в зависимости от источника привлечения

# In[274]:


plot_retention(retention_grouped, retention_history, horizon_days)


# Удержание у каналов FaceBoom и AdNonSense гораздо ниже, чем у остальных. Удержание у канала TipTop мало отличается от остальных источников, скорее всего причина неокупаемости - высокая стоимость привлечения.

# ### Напишите выводы
# 
# - Выделите причины неэффективности привлечения пользователей.
# - Сформулируйте рекомендации для отдела маркетинга.

# В ходе анализа было выявлено:
# - Больше всего пользователей из США, меньше всего из Германии. Больше всего платящих пользователей в США, меньше всего - во Франции
# - Клиенты чаще пользуются iPhone (54479), больше всего платящих пользователей также оттуда, реже всего - Мас (30042). Меньше платящих пользователей приходит из РС
# - Самый популярный канал - FaceBoom(56439), наименее популярный - lambdaMediaAds(2149). Больше всего платящих пользователей пришло из канала FaceBoom, меньше - из OppleCreativeMedia
# 
# Была проанализирована окупаемость рекламы:
# - Посчитана общая сумма расходов на маркетинг - 105497
# - Больше всего денег было потрачено на источник TipTop (54751), меньше всего (если не учитывать organic, т.к. там будет 0) - YRabbit (944)
# - У большинства источников САС варьировалась в пределах от 0 до 50 долларов, однако у двух источников (TipTop и FaceBoom) показатель превысил 100 долларов, а в начале октября 2019 САС TipTop достиг 600 долларов
# - В динамике расходов по неделям выдеяются два источника - TipTop и FaceBoom, у обоих источников был пик по расходам на 39 неделе
# - Пик расходов по месяцам у двух самых затратных источников: TipTop - сентябрь, FaceBoom - август
# - Лидер по стоимости привлечения одного пользователя - TipTop (около 3.5 долларов)
# - Лидер по стоимости привлечения одного пользователя по странам - США
# 
# Среди причин неэффективности привлечения пользователей могут быть слишком большие траты на некоторые источники:
# - Источники FaceBoom и AdNonSense являются неэффективными - они не окупают себя и обладают рекордно низким удержанием, поэ

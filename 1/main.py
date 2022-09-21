import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import *

global select
select = 20


# Вывод выборки

def selectionPrint():
    print('Выборка: ')
    for i in range(1, select + 1):
        print(i, end=' ')
    print('\n')


selectionPrint()

###############
N = int(input())
list1 = []
###############

# Чтение файла в зависимости от N(номера по списку)

def readTXT():
    global line
    f = open('numbers.txt', 'r')

    if N == 3:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд Волкова Данила')
    elif N == 2:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд Войтова Никиты')
    elif N == 1:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд Вахрушевой Алины')
    elif N == 4:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд Горюнова Алексея')
    elif N == 7:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд Железняка Кирилла')
    else:
        line = f.read().split('\n')[N - 1]
        print('Вы выбрали ряд человека не из нашей команды, но я все равно выведу вам его данные')


readTXT()


# Раскладка по спискам

def listSort():
    global list1
    for char in list(line):
        if char != ' ':
            if char != '\n':

                if char == '.':
                    print('\n', char, 'Float не допустимо')
                    exit()
                else:
                    try:
                        tmp = int(char)
                    except:
                        print('\n', char, 'Не является числом!')
                        exit()

                    print(char, end=' ')
                    list1.append(int(char))


listSort()

print('\n')
print('1 ЗАДАНИЕ\n'
      '___________________________________________')
print()


# Мода

def modaFind():
    global moda
    unique_char = set(list1)
    moda = []

    for uni_char in unique_char:
        count = list1.count(uni_char)
        moda.append([uni_char, count])

    # print(moda)
    maxM = 0
    for i in moda:
        if i[1] > maxM:
            maxM = i[1]
            nametmp = i[0]
    print(f'Мода: {nametmp}')


modaFind()


# Выбор среднее

def mathMeanFind():
    global mathWait
    mathWait = (moda[0][0] * moda[0][1] + moda[1][0] * moda[1][1] + moda[2][0] * moda[2][1] + moda[3][0] \
                * moda[3][1] + moda[4][0] * moda[4][1] + moda[5][0] * moda[5][1]) / select
    print(f'Мат ожидание: {mathWait}')


mathMeanFind()


# Медиана
def medianFind():
    listtmp = list1
    listtmp.sort()

    median = (listtmp[9] + listtmp[10]) / 2
    print(f'Медиана: {median}')


medianFind()


# Дисперсия

def dispFind():
    global disp
    disp = (((moda[0][0] - mathWait) ** 2 * moda[0][1]) + ((moda[1][0] - mathWait) ** 2 * moda[1][1]) + (
            (moda[2][0] - mathWait) ** 2 * moda[2][1]) \
            + ((moda[3][0] - mathWait) ** 2 * moda[3][1]) + ((moda[4][0] - mathWait) ** 2 * moda[4][1]) + (
                    (moda[5][0] - mathWait) ** 2 * moda[5][1])) / (select - 1)
    print(f'Дисперсия: {round(disp, 4)}')


dispFind()


# Отклонение

def varianceFind():
    global variance
    variance = sqrt((((moda[0][0] - mathWait) ** 2 * moda[0][1]) + ((moda[1][0] - mathWait) ** 2 * moda[1][1]) + (
            (moda[2][0] - mathWait) ** 2 * moda[2][1]) \
                     + ((moda[3][0] - mathWait) ** 2 * moda[3][1]) + ((moda[4][0] - mathWait) ** 2 * moda[4][1]) + (
                             (moda[5][0] - mathWait) ** 2 * moda[5][1])) / (select - 1))
    print(f'Отклонение: {round(variance, 4)}')


varianceFind()


# Коеффициент вариации

def coefVarianceFind():
    coef = (variance * 100) / mathWait
    print(f'Коеффициент вариации: {round(coef, 4)}')


coefVarianceFind()


# Размах

def rangeFind():
    global max
    global min
    max = max(list1)
    min = min(list1)

    range = max - min
    print(f'Размах: {range}')


rangeFind()


# Мода отрисовка

def modaViewGraphs():
    fig = plt.figure()

    x = list1
    n, bins, patches = plt.hist(x, 30, facecolor='green', alpha=0.5)

    graph1 = plt.plot([moda[0][0], moda[0][0]], [moda[0][1], moda[0][1]])
    graph1 = plt.plot([moda[0][0], moda[0][0] + 1], [moda[0][1], moda[1][1]])
    graph1 = plt.plot([moda[0][0] + 1, moda[0][0] + 2], [moda[1][1], moda[2][1]], 'go-', label='line 1', linewidth=2)
    graph1 = plt.plot([moda[0][0] + 2, moda[0][0] + 3], [moda[2][1], moda[3][1]], 'go-', label='line 1', linewidth=2)
    graph1 = plt.plot([moda[0][0] + 3, moda[0][0] + 4], [moda[3][1], moda[4][1]], 'go-', label='line 1', linewidth=2)
    graph1 = plt.plot([moda[0][0] + 4, moda[0][0] + 5], [moda[4][1], moda[5][1]], 'go-', label='line 1', linewidth=2)
    grid1 = plt.grid(True)
    plt.show()


print('\n')
print('2 ЗАДАНИЕ\n'
      '___________________________________________')
print()


def meanProfitFind():
    meanIncome = 3500 / 1000

    meanProfitTMP1 = mathWait - variance
    meanProfitTMP2 = mathWait + variance

    result2 = False

    if meanProfitTMP1 < meanIncome < meanProfitTMP2:
        result2 = True
    else:
        result2 = False

    print(f'{round(meanProfitTMP1, 2)} < {meanIncome} < {round(meanProfitTMP2, 2)}')
    if result2:
        print('Неравнество ВЕРНО, прибыль равна предполагаемой!')
    else:
        print('Неравнество НЕВЕРНО, прибыль НЕ равна предполагаемой!')


meanProfitFind()
modaViewGraphs()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import *

select = 20

#Вывод выборки

print('Выборка: ')
for i in range(1, select + 1):
    print(i, end=' ')
print('\n')

N = int(input())

list1 = []

#Открывание файла в зависимости от N

f = open('numbers.txt', 'r')
if N == 3:
    line = f.readline()
if N == 22:
    line = f.read().split('\n')[1]

#Раскладка по спискам

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
print('\n')

#Мода

unique_char = set(list1)
moda = []

for uni_char in unique_char:
    count = list1.count(uni_char)
    moda.append([uni_char, count])

print(moda)
maxM = 0
for i in moda:
    if i[1] > maxM:
        maxM = i[1]
        nametmp = i[0]
print(f'Мода: {nametmp}')

#Выбор среднее

mathWait = (moda[0][0] * moda[0][1] + moda[1][0] * moda[1][1] + moda[2][0] * moda[2][1] + moda[3][0] \
           * moda[3][1] + moda[4][0] * moda[4][1] + moda[5][0] * moda[5][1]) / select
print(f'Мат ожидание: {mathWait}')

#Медиана

median = (list1[9] + list1[10]) / 2
print(f'Медиана: {median}')

#Дисперсия

disp = (((moda[0][0] - mathWait)**2 * moda[0][1]) + ((moda[1][0] - mathWait)**2 * moda[1][1])+ ((moda[2][0] - mathWait)**2 * moda[2][1]) \
       + ((moda[3][0] - mathWait)**2 * moda[3][1]) + ((moda[4][0] - mathWait)**2 * moda[4][1]) + ((moda[5][0] - mathWait)**2 * moda[5][1])) / (select - 1)
print(f'Дисперсия: {round(disp, 4)}')

#Отклонение

variance = sqrt((((moda[0][0] - mathWait)**2 * moda[0][1]) + ((moda[1][0] - mathWait)**2 * moda[1][1])+ ((moda[2][0] - mathWait)**2 * moda[2][1]) \
       + ((moda[3][0] - mathWait)**2 * moda[3][1]) + ((moda[4][0] - mathWait)**2 * moda[4][1]) + ((moda[5][0] - mathWait)**2 * moda[5][1])) / (select - 1))
print(f'Отклонение: {round(variance, 4)}')

#Коеффициент вариации

coef = (disp * 100) / mathWait
print(f'Коеффициент вариации: {round(coef, 4)}')

#Размах

max = max(list1)
min = min(list1)

range = max - min
print(f'Размах: {range}')


#Мода отрисовка

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


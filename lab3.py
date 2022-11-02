import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import RadioButtons
from sklearn.linear_model import LinearRegression
from matplotlib.artist import Artist
import seaborn as sns


# Параметры плота и графиков

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams.update({'font.size': 7})
plt.style.use('bmh')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Импорт эксель, создание дата фрейма

exl = pd.read_excel('sheetEmpty.xlsx',index_col=0, na_filter=1)
fig, ax = plt.subplots(1, figsize=(13,7))
plt.subplots_adjust(left=0.3)
exlData = pd.DataFrame(exl)

# x - координата

x = ('январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь'
            )

# Na -> 0
exlData = exlData.fillna(0)

# Заполнение пропусков
for year in range(0,6):
    i = 0
    for na in exlData.iloc[:, year].values:
        if na < 0.1:
            minus = exlData.iloc[:, year+1].values[0] - exlData.iloc[:, year].values[0]
            procent = minus / exlData.iloc[:, year+1].values[0] * 100
            tmp = exlData.iloc[:, year+1].values[i] / 100 * procent
            na = exlData.iloc[:, year+1].values[i] - tmp
            na = round(na,0)
            exlData.iloc[:, year].values[i] = na
        i += 1

# Предсказание 2023 и 2024
month = 0
for khk in range(0,12):
    middleX = (2017 + 2018 + 2019 + 2020 + 2021 + 2022) / 6 # среднее годов
    middleY = (exlData.iloc[month, :].values[0] + exlData.iloc[month, :].values[1] + exlData.iloc[month, :].values[2] + \
    exlData.iloc[month, :].values[3] + exlData.iloc[month, :].values[4] + exlData.iloc[month, :].values[5]) / 6 # среднее значений
    b = (((2017 - middleX) * (exlData.iloc[month, :].values[0] - middleY)) + ((2018 - middleX) * (exlData.iloc[month, :].values[1] - middleY)) + \
         ((2019 - middleX) * (exlData.iloc[month, :].values[2] - middleY)) +((2020 - middleX) * (exlData.iloc[month, :].values[3] - middleY)) + \
         ((2021 - middleX) * (exlData.iloc[month, :].values[4] - middleY) +(2022 - middleX) * (exlData.iloc[month, :].values[5] - middleY)))  \
        / (((2017 - middleX)**2) + ((2018 - middleX)**2) + ((2019 - middleX)**2) + ((2020 - middleX)**2) + ((2021 - middleX)**2) + ((2022 - middleX)**2))
    a = middleY - b * middleX
    res = a + b * 2023
    res = round(res,0)
    exlData.iloc[:, 6].values[month] = res
    month += 1
    # y = a + bx

jj = 2000
jj2 = 3000

month = 0
for khk in range(0,12):
    middleX = (2017 + 2018 + 2019 + 2020 + 2021 + 2022 + 2023) / 7 # среднее годов
    middleY = (exlData.iloc[month, :].values[0] + exlData.iloc[month, :].values[1] + exlData.iloc[month, :].values[2] + \
    exlData.iloc[month, :].values[3] + exlData.iloc[month, :].values[4] + exlData.iloc[month, :].values[5] + exlData.iloc[month, :].values[6])/ 7 # среднее значений
    b = (((2017 - middleX) * (exlData.iloc[month, :].values[0] - middleY)) + ((2018 - middleX) * (exlData.iloc[month, :].values[1] - middleY)) + \
         ((2019 - middleX) * (exlData.iloc[month, :].values[2] - middleY)) + ((2020 - middleX) * (exlData.iloc[month, :].values[3] - middleY)) + \
         ((2021 - middleX) * (exlData.iloc[month, :].values[4] - middleY) + (2022 - middleX) * (exlData.iloc[month, :].values[5] - middleY)) + \
         (2023 - middleX) * (exlData.iloc[month, :].values[6] - middleY)) \
        / (((2017 - middleX)) ** 2 + ((2018 - middleX) ** 2) + ((2019 - middleX) ** 2) + ((2020 - middleX) ** 2) + ((2021 - middleX) ** 2) + ((2022 - middleX) ** 2) + \
           ((2023 - middleX) ** 2))
    a = middleY - b * middleX
    res = a + b * 2024
    res = round(res,0)
    exlData.iloc[:, 7].values[month] = res
    month += 1
    # y = a + bx

txt1,txt2,txt3 = 'a','a','a'

# Создание Y координат для линейной регрессии и графиков
s1 = exlData.iloc[:, 0].values
s2 = exlData.iloc[:, 1].values
s3 = exlData.iloc[:, 2].values
s4 = exlData.iloc[:, 3].values
s5 = exlData.iloc[:, 4].values
s6 = exlData.iloc[:, 5].values
s7 = exlData.iloc[:, 6].values
s8 = exlData.iloc[:, 7].values
lr2 = exlData.iloc[2:5, 0:6].values
lr3 = exlData.iloc[6:9, 0:6].values
lr4 = exlData.iloc[7:10, 0:6].values
lr = np.array([exlData.iloc[11, 0:6].values, exlData.iloc[0, 0:6].values, exlData.iloc[1, 0:6].values])

model = LinearRegression()

# очистка вывода инфы на регрессии
def dltText():
    if txt1 != 'a':
        try:
            Artist.remove(txt1)
            Artist.remove(txt2)
            Artist.remove(txt3)
        except:
            return

# коеффициент оценки
def coef(x,y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1*mean_x
    return (b_0, b_1)

# высчитывание предсказания значений, отрисовка регрессии
def plot_progression_line(x, y, b, label):
    global txt1,txt2,txt3
    txt1,txt2,txt3 = 'a','a','a'
    model.fit(x,y)
    r_sq = model.score(x,y)
    y_pred = b[0] + b[1] * x
    txt1 = plt.figtext(0.01, 0.50, ('coefficient of determination: ',r_sq), size=9, c='gray')
    txt2 = plt.figtext(0.01, 0.04, ('intercept:', model.intercept_), size=9, c='gray')
    txt3 = plt.figtext(0.01, 0.15, ('slope:', model.coef_), size=7, c='gray')
    ax.scatter(x, y, color='gray')
    ax.plot(x, y_pred, 'k--')

# Отрисовка плото и графиков по нажатию на кнопки
def click(label):
    ax.clear()
    if label == "2017":
        dltText()
        ax.plot(x, s1, lw=2, color='dodgerblue')
        ax.set_title("2017", c='dodgerblue')
    elif label == "Все":
        dltText()
        ax.set_title("2017-2023")
        ax.plot(exlData)
    elif label == "2018":
        dltText()
        ax.plot(x, s2, lw=2, color='firebrick')
        ax.set_title("2018", c='firebrick')
    elif label == "2019":
        dltText()
        ax.plot(x, s3, lw=2, color='mediumpurple')
        ax.set_title("2019", c='mediumpurple')
    elif label == "2020":
        dltText()
        ax.plot(x, s4, lw=2, color='seagreen')
        ax.set_title("2020", c='seagreen')
    elif label == "2021":
        dltText()
        ax.plot(x, s5, lw=2, color='coral')
        ax.set_title("2021", c='coral')
    elif label == "2022":
        dltText()
        ax.plot(x, s6, lw=2, color='lightpink')
        ax.set_title("2022", c='lightpink')
    elif label == "2023":
        dltText()
        ax.plot(x, s7, lw=2, color='aqua')
        ax.set_title("2023", c='aqua')
    elif label == "2024":
        dltText()
        ax.plot(x, s8, lw=2, color='mediumspringgreen')
        ax.set_title("2024", c='mediumspringgreen')

    elif label == "Linear Regression SEAS1":
        mon = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]])
        dltText()
        b = coef(mon, lr)
        plot_progression_line(mon,lr+jj,b,label)
        ax.set_title(label, c='red')
        ax.set_xlabel('Декабрь(1), Январь(2), Февраль(3)')
    elif label == "Linear Regression SEAS2":
        mon = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]])
        dltText()
        b = coef(mon, lr2)
        plot_progression_line(mon,lr2,b, label)
        ax.set_title(label, c='red')
        ax.set_xlabel('Март(1), Апрель(2), Май(3)')
    elif label == "Linear Regression SEAS3":
        mon = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]])
        dltText()
        b = coef(mon, lr3)
        plot_progression_line(mon,lr3,b, label)
        ax.set_title(label, c='red')
        ax.set_xlabel('Июнь(1), Июль(2), Август(3)')
    elif label == "Linear Regression SEAS4":
        mon = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]])
        dltText()
        b = coef(mon, lr4)
        plot_progression_line(mon, lr4+jj2, b, label)
        ax.set_title(label, c='red')
        ax.set_xlabel('Сентябрь(1), Октябрь(2), Ноябрь(3)')
    plt.draw()

print(exlData)

# кнопочки
rax = plt.axes([0.02, 0.55, 0.20, 0.35], facecolor='white')
radio = RadioButtons(rax, ('Все', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'Linear Regression SEAS1', 'Linear Regression SEAS2', 'Linear Regression SEAS3', 'Linear Regression SEAS4'), activecolor='k')
plt.title(r'Доход магазинчкика')

# текст для различения графиков на общем плоте
plt.figtext(0.93, 0.65, '2024', size=12, c='mediumspringgreen')
plt.figtext(0.93, 0.60, '2023', size=12, c='aqua')
plt.figtext(0.93, 0.55, '2022', size=12, c='lightpink')
plt.figtext(0.93, 0.50, '2021', size=12, c='coral')
plt.figtext(0.93, 0.45, '2020', size=12, c='seagreen')
plt.figtext(0.93, 0.40, '2019', size=12, c='mediumpurple')
plt.figtext(0.93, 0.35, '2018', size=12, c='firebrick')
plt.figtext(0.93, 0.30, '2017', size=12, c='dodgerblue')

# начальный запуск
ax.plot(exlData)
ax.set_title("2017-2024")
radio.on_clicked(click)

plt.show()

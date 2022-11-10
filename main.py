import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
import sklearn.metrics
from matplotlib.widgets import RadioButtons
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant




pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 150)
fig, ax = plt.subplots(1, figsize=(13,7))
plt.subplots_adjust(left=0.3)

model = LinearRegression()


exl = pd.read_csv ('mcs_ds_edited_iter_shuffled.csv')
exlData = pd.DataFrame(exl)

exlData = exlData.drop(exlData.columns[[5]],axis=1)
x = exlData.iloc[:, 4].values


def plot_progression_line(x, y, b):
    model.fit(x,y)
    r_sq = model.score(x,y)
    y_pred = b[0] + b[1] * x
    ax.scatter(x, y, color='gray')
    ax.plot(x, y_pred, 'k--')

def coef(x,y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1*mean_x
    return (b_0, b_1)

def click(label):
    ax.clear()
    if label == "linear Regression of trans_range":
        ax.scatter(x, y2, color='gray')
        ax.plot([np.min(x), np.max(x)], [np.min(y2), np.max(y2)], 'k--')
        ax.set_title("linear Regression of trans_range", c='dodgerblue')
    elif label == "linear Regression of anchor_ratio":
        ax.scatter(x, y1, color='gray')
        ax.plot([np.min(x), np.max(x)], [np.min(y1), np.max(y1)], 'k--')
        ax.set_title("linear Regression of anchor_ratio", c='dodgerblue')
    elif label == "linear Regression of node_density":
        ax.scatter(x, y3, color='gray')
        ax.plot([np.min(x), np.max(x)], [np.min(y3), np.max(y3)], 'k--')
        ax.set_title("linear Regression of node_density", c='dodgerblue')
    elif label == "linear Regression of iterations":
        ax.scatter(x, y4, color='gray')
        ax.plot([np.min(x), np.max(x)], [np.min(y4), np.max(y4)], 'k--')
        ax.set_title("linear Regression of iterations", c='dodgerblue')


def compute_vif(considered_features):
    X = exlData[considered_features]
    X['intercept'] = 1

    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif

considered_features =['anchor_ratio', 'trans_range', 'node_density', 'iterations']
a = compute_vif(considered_features).sort_values('VIF', ascending=False)
print(a)

y1 = exlData.iloc[:,0].values
y2 = exlData.iloc[:,1].values
y3 = exlData.iloc[:,2].values
y4 = exlData.iloc[:,3].values
y5 = exlData.iloc[:,4].values

rax = plt.axes([0.02, 0.55, 0.20, 0.35], facecolor='white')
radio = RadioButtons(rax, ('linear Regression of anchor_ratio', 'linear Regression of trans_range', 'linear Regression of node_density',
                           'linear Regression of iterations'), activecolor='k')

x2 = np.c_[exlData['anchor_ratio'],exlData['trans_range'],exlData['node_density'],exlData['iterations']]
X_train, X_test, y_train, y_test = train_test_split(x2, y5, test_size=0.4)
model.fit(X_train,y_train)
pred = model.predict(X_test)
y_pred = model.predict(x2)
test_rmse = (np.sqrt(mean_squared_error(y_test,pred)))
testR2 = sklearn.metrics.r2_score(y_test, pred)
print('mean_squared_error: ',mean_squared_error(y5, y_pred))
print('mean_accuracy_error ',model.score(X_test, y_test))
print('показатели точности: ')
print(test_rmse)
print(testR2)
print('intercept: ', model.intercept_)
print('score: ', model.score(x2,y5))
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
print('coef: ', model2.feature_importances_)


plt.figure(figsize=(9,6))
mask = np.zeros_like(exlData.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
sns.heatmap(exlData.corr(), mask=mask, annot=True, annot_kws={'size': 14})
sns.set_style('white')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

rm_tgt_corr=round(exlData['anchor_ratio'].corr(exlData['trans_range']), 3)
plt.figure(figsize=(9,6))
plt.scatter(x=exlData['anchor_ratio'], y=exlData['trans_range'], alpha=0.6, s=80, color='blue')
plt.xlabel('anchor_ratio')
plt.ylabel('trans_range')

ax.scatter(x, y1, color='gray')
ax.plot([np.min(x), np.max(x)], [np.min(y1), np.max(y1)], 'k--')
ax.set_title("linear Regression of anchor_ratio", c='dodgerblue')
radio.on_clicked(click)
print(exlData)
plt.show()

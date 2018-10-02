
Андрей Сенаев РИМ-181228

# Линейная регрессия

Задача прогноза вещественного признака по прочим признакам (задача восстановления регрессии) решается минимизацией квадратичной ошибки. Рассмотрим линейную регресси на примере листка ириса. Будем предсказывать длину листка по его ширине. x - ширина листа, y - длина листка.


```python
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
data = load_iris().data
x = data[:,3]
y = data[:,2]
plt.scatter(x, y)
plt.show()
```


    <Figure size 640x480 with 1 Axes>


Напишите функцию, которая по двум параметрам  $w_0$ и $w_1$ вычисляет квадратичную ошибку приближения зависимости роста $y$ от веса $x$ прямой линией $y = w_0 + w_1 * x$:

$$Err(w_0, w_1) = \frac{1}{n}\sum_{i=1}^n {(y_i - (w_0 + w_1 * x_i))}^2 $$ Здесь $n$ – число наблюдений в наборе данных, $y_i$ и $x_i$ – рост и вес $i$-ого человека в наборе данных.


```python
def squarErr(w0, w1):
    squareDeviationSum = .0;
    for n, a in enumerate(data):
        xi = a[3]
        yi = a[2]
        
        squareDeviation = (yi - (w0 + w1 * xi)) ** 2
        
        squareDeviationSum = squareDeviationSum + squareDeviation
    return squareDeviationSum
```

Возьмите параметры $\omega_0$ - свободный член и $\omega_1$ - наклон прямой и постройте две любые прямые, которые быдут некоторым образом описывать зависмость ширины листа от его длины. Представьте графически.


```python
lineFunc = lambda x, w0, w1: w0 + w1 * x

def showLambda (lambdaFunc):
    xpts = np.linspace(-1, 4, 500)
    test_v = np.vectorize(lambdaFunc)
    plt.plot(xpts, test_v(xpts))
    plt.scatter(x, y)
    plt.show()

abscessFunc = lambda x: 0 + 0 * x
otherFunc = lambda x: -1 + 1 * x

print(y)

print('квадратичное (не средне) отклонение от оси абсцисс', squarErr(0, 0))
showLambda(abscessFunc)

print('квадратичное (не средне) отклонение от нечетной функции под сорокпять градусов', squarErr(0, 1))
showLambda(otherFunc)
```

    [1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 1.5 1.6 1.4 1.1 1.2 1.5 1.3 1.4
     1.7 1.5 1.7 1.5 1.  1.7 1.9 1.6 1.6 1.5 1.4 1.6 1.6 1.5 1.5 1.4 1.5 1.2
     1.3 1.5 1.3 1.5 1.3 1.3 1.3 1.6 1.9 1.4 1.6 1.4 1.5 1.4 4.7 4.5 4.9 4.
     4.6 4.5 4.7 3.3 4.6 3.9 3.5 4.2 4.  4.7 3.6 4.4 4.5 4.1 4.5 3.9 4.8 4.
     4.9 4.7 4.3 4.4 4.8 5.  4.5 3.5 3.8 3.7 3.9 5.1 4.5 4.5 4.7 4.4 4.1 4.
     4.4 4.6 4.  3.3 4.2 4.2 4.2 4.3 3.  4.1 6.  5.1 5.9 5.6 5.8 6.6 4.5 6.3
     5.8 6.1 5.1 5.3 5.5 5.  5.1 5.3 5.5 6.7 6.9 5.  5.7 4.9 6.7 4.9 5.7 6.
     4.8 4.9 5.6 5.8 6.1 6.4 5.6 5.1 5.6 6.1 5.6 5.5 4.8 5.4 5.6 5.1 5.1 5.9
     5.7 5.2 5.  5.2 5.4 5.1]
    квадратичное (не средне) отклонение от оси абсцисс 2583.0000000000005



![png](output_6_1.png)


    квадратичное (не средне) отклонение от нечетной функции под сорокпять градусов 1147.3600000000004



![png](output_6_3.png)


Минимизация квадратичной функции ошибки - относительная простая задача, поскольку функция выпуклая. Для такой задачи существует много методов оптимизации. Рассмотрим, как функция ошибки зависит от одного параметра (наклон прямой), если второй параметр (свободный член) зафиксировать.

Постройте график зависимости функции ошибки от параметра $w_1$ при $w_0$ = 0.


```python
defiationForW1Lambda = lambda x: squarErr(0, x);

deviationsForW1 = np.array([[key / 100, defiationForW1Lambda(key / 100)] for key in range(-1000, 1001)])

plt.plot(deviationsForW1[:,0], deviationsForW1[:,1], linewidth=2.0)
plt.show()
```


![png](output_9_0.png)


С помощью метода minimize_scalar из scipy.optimize найдите минимум функции, определенной выше, для значений параметра  $w_1$ в диапазоне [-10,10]. Проведите на графике прямую, соответствующую значениям параметров ($w_0$, $w_1$) = (0, $w_1\_opt$), где $w_1\_opt$ – найденное в оптимальное значение параметра $w_1$.


```python
from scipy.optimize import minimize_scalar

minimize_scalar(defiationForW1Lambda, bounds=(-10, 10))
```




         fun: 85.1208703274892
        nfev: 10
         nit: 4
     success: True
           x: 2.8745286139596438



В связи с тем, что у нас требуется найти минимальное значение функции по $w_0$, $w_1$ следовательно функция ошибки будет находиться в трехмерном пространстве.
Пример построения трехмерных графиков находится ниже.


```python
from mpl_toolkits.mplot3d import Axes3D
```


```python
fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X + Y)


surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```


![png](output_14_0.png)


Постройте график зависимости функции ошибки в трехмерном пространстве от параметров $\omega_0$ и $\omega_1$.


```python
figire = plt.figure()
axis = figire.gca(projection='3d') # get current axis

w0 = np.arange(-1000, 1000, 10)
w1 = np.arange(-10, 10, 0.1)
w0, w1 = np.meshgrid(w0, w1)
deviation = squarErr(w0, w1)


surface = axis.plot_surface(w0, w1, deviation)
axis.set_xlabel('w0')
axis.set_ylabel('w1')
axis.set_zlabel('deviation')
plt.show()
```


![png](output_16_0.png)


Используя метод minimize найдите минимум функции. Диапазон поиска подберите самостоятельно. Начальная точка - (0,0). Постройте прямую на графике с данными.


```python
from scipy.optimize import minimize

def calculateSquareError(data):
    return squarErr(data[1], data[0])

[w1min, w0min] = minimize(calculateSquareError, [.0, .0], method='L-BFGS-B', bounds=([-10, 10], [-1000, 1000]))['x']

plt.scatter(x, y)

minFunc = lambda x: w0min + w1min * x;

minFuncValues = np.array([[key / 1000, minFunc(key / 1000)] for key in range(0, 3000)])

plt.plot(minFuncValues[:,0], minFuncValues[:,1], linewidth=2.0)

print('w0 minimum', w0min)
print('w1 minimum', w1min)

plt.show()
```

    w0 minimum 1.0905721871207217
    w1 minimum 2.225885276027521



![png](output_18_1.png)


Проверьте полученные результаты аналитическим методом поиска корней СЛАУ использованным на лекции.


```python
A = np.vstack([x, np.ones(len(x))]).T

m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)
```

    2.225885306553912 1.0905721458773783


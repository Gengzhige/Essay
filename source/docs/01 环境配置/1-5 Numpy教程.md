## 1-5 Numpy教程

### 1. Numpy是什么？

**Numerical Python** 的缩写

- 一个开源的Python科学计算库
- 使用numpy可以方便的使用数组、矩阵进行计算
- 包含线性代数、傅里叶变换、随机数生成等大量函数

### 2. 为什么使用Numpy？
对于同样的数值计算任务，使用numpy比直接编写python代码实现 优点：
- 代码更简洁： numpy直接以数组、矩阵为粒度计算并且支持大量的数学函数，而python需要用for循环从底层实现；
- 性能更高效： numpy的数组存储效率和输入输出计算性能，比python使用list好很多；numpy的大部分代码都是c语言实现的，这是numpy比python高效的原因

numpy是python各种数据科学类库的基础库
- 比如SciPy、Scikit-Learn、tenorflow、paddlepaddle等
- 如果不会numpy，这些库的深入理解都会遇到障碍


### 3. 安装numpy
conda install numpy

(关于conda的使用可以看上期教程)

### 4. numpy 与 原生python的性能对比



```python
# 引入numpy的包 
import numpy as np
```


```python
# 查看numpy的版本
np.__version__
```


```python
# 使用python原生语法
# 实现两个数组相加
def python_sum(n):
    a = [i**2 for i in range(n)]
    b = [i**3 for i in range(n)]
    c = [a[i]+b[i] for i in range(n)]
    return c
    
```


```python
# 测试
python_sum(10)
```




>    [0, 2, 12, 36, 80, 150, 252, 392, 576, 810]




```python
# 使用numpy实现上面功能
def numpy_sum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a+b
    return c
```


```python
# 测试
numpy_sum(10)
```




>    array([  0,   2,  12,  36,  80, 150, 252, 392, 576, 810], dtype=int32)




```python
# 执行1000次  输入执行时间 对比性能
# %timeit 魔法函数 
%timeit python_sum(1000)
```

>    411 µs ± 3.31 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    


```python
%timeit numpy_sum(1000)
```

>    6.51 µs ± 57 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    

**可以看出使用numpy进行计算要比原生Python快得多，而且数据量越大，效果越明显**

### 5.numpy核心ndarray对象 

#### ndarray对象：
- numpy的核心数据结构，叫做array就是数组，array对象可以是一维数组，也可以是多维数组
- python的list也可以实现相同的功能，但是array的优势在于性能好，包含数组元数据信息、大量的便捷函数
- 成为 Scipy、Pandas、scilit-learn、tensorflow、paddlepaddle等框架的通用底层语言
- numpy的array和python的list的一个区别是它的元素必须都是同一种数据类型，这也是numpy高性能的一个原因

#### ndarray属性
- shape 返回一个元组 表示array的形状
- ndim 返回一个数字 表示array的维度的数目
- size 返回一个数字 表示array中所有数据元素的数目
- dtype array中元素的数据类型
- itemsize 表示数组中每个元素的字节大小

#### array本身支持的大量操作和函数
- 直接逐元素的加减乘除等算数操作
- 更好用的面向多维的数组索引
- 求sum/mean等聚合函数
- 线性代数函数、比如求解逆矩阵、求解方程组


### 6. 创建array的方法
- 从Python的列表list和嵌套列表创建array
- 使用预定函数arange、linspace等创建等差数组
- 使用ones、ones_like、zeros、zeros_like、empty、empty_like、full、full_like、eye等函数创建
- 生成随机数的np.random模块创建


#### 6.1 使用np.array()


```python
import numpy as np
# 创建一个一维数组
x = np.array([1, 2, 3 ,4, 5])
x
```




>    array([1, 2, 3, 4, 5])




```python
# 创建一个二维数组
y = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7 ,8]
    ]
)
y
```




>    array([[1, 2, 3, 4],
>           [5, 6, 7, 8]])




```python
# 数组的形状  1行5列
x.shape
```




>    (5,)




```python
# 数组的形状  2行4列
y.shape
```




>    (2, 4)




```python
# x是1维数组
x.ndim
```




>    1




```python
# y是2维数组
y.ndim
```




>    2




```python
# x一共有5个数据
x.size
```




>    5




```python
# y中一共有8个数据
y.size
```




>    8




```python
# x中的数据类型
x.dtype
```




>    dtype('int32')




```python
# y中的数据类型
y.dtype
```




>    dtype('int32')




```python
# 表示数组中每个元素的字节大小
y.itemsize
```




>    4



#### 6.2 使用np.arange(start,stop,step,dtype) 生成等差数组
- start 表示开始的数（包含） 默认从0开始
- stop 表示结束的数（不包含）
- step 指定步长   默认为1
- dtype 指定数据类型


```python
 # 创建了一个从0到9的数组 不包含10  步长默认为1
np.arange(10)
```




>    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# [0,10)区间  左闭右开   指定步长为2
np.arange(0, 10, 2)
```




>    array([0, 2, 4, 6, 8])




```python
# reshape 可以改变数组的形状
# 将一维数组 改成 2行5列的 2维数组
np.arange(10).reshape(2,5)
```




>    array([[0, 1, 2, 3, 4],
>           [5, 6, 7, 8, 9]])



#### 6.3 使用np.linspace(start,stop,num,endpoint)  创建等差数组-指定数量
- start 起始值
- stop 结束值
- num 要生成的等间隔样例数量，默认为50
- endpoint 序列中是否包含stop值， 默认为true


```python
np.linspace(0,10)
```




>    array([ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,  0.81632653,
>            1.02040816,  1.2244898 ,  1.42857143,  1.63265306,  1.83673469,
>            2.04081633,  2.24489796,  2.44897959,  2.65306122,  2.85714286,
>            3.06122449,  3.26530612,  3.46938776,  3.67346939,  3.87755102,
>            4.08163265,  4.28571429,  4.48979592,  4.69387755,  4.89795918,
>            5.10204082,  5.30612245,  5.51020408,  5.71428571,  5.91836735,
>            6.12244898,  6.32653061,  6.53061224,  6.73469388,  6.93877551,
>            7.14285714,  7.34693878,  7.55102041,  7.75510204,  7.95918367,
>            8.16326531,  8.36734694,  8.57142857,  8.7755102 ,  8.97959184,
>            9.18367347,  9.3877551 ,  9.59183673,  9.79591837, 10.        ])




```python
np.linspace(0, 10, 5)
```




>    array([ 0. ,  2.5,  5. ,  7.5, 10. ])



#### 6.4 使用ones、ones_like、zeros、zeros_like、empty、empty_like、full、full_like、eye等函数创建


```python
# 使用ones创建全是1的数组
np.ones(10)
```




>    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
np.ones((2, 4))
```




>    array([[1., 1., 1., 1.],
>           [1., 1., 1., 1.]])




```python
# 使用ones_like创建形状相同的数组
np.ones_like(x)
```




>    array([1, 1, 1, 1, 1])




```python
np.ones_like(y)
```




>    array([[1, 1, 1, 1],
>           [1, 1, 1, 1]])




```python
# 使用zeros创建全是0的数组
np.zeros(10)
```




>    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.zeros((2,4))
```




>    array([[0., 0., 0., 0.],
>           [0., 0., 0., 0.]])




```python
# 使用zeros_like创建形状相同的数组
np.zeros_like(x)
```




>    array([0, 0, 0, 0, 0])




```python
np.zeros_like(y)
```




>    array([[0, 0, 0, 0],
>           [0, 0, 0, 0]])




```python
# 使用empty创建未初始化的数组
# 注意：数据是未初始化的，里面的值可能是随机值不要用
np.empty(10)

```




>    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.empty((2, 3))
```




>    array([[0., 0., 0.],
>           [0., 0., 0.]])




```python
# 使用empty_like创建形状相同的数组
np.empty_like(x)
```




>    array([1767994469,  170816364,  538976266, 1701139232, 1936474400])




```python
np.empty_like(y)
```




>    array([[  24385651,    8127092,   60818051,   27328636],
>           [   6226044,    6946940,   19464275, -637261490]])




```python
# 使用full创建指定值的数组
np.full(10, 666)
```




>    array([666, 666, 666, 666, 666, 666, 666, 666, 666, 666])




```python
np.full((2, 4), 666)
```




>    array([[666, 666, 666, 666],
>           [666, 666, 666, 666]])




```python
# 使用full_like创建形状相同的数组
np.full_like(x, 666)
```




>    array([666, 666, 666, 666, 666])




```python
np.full_like(y, 666)
```




>    array([[666, 666, 666, 666],
>           [666, 666, 666, 666]])




```python
# 使用 eye 函数创建 对角线的地方为1，其余的地方为0.
np.eye(5)
```




>    array([[1., 0., 0., 0., 0.],
>           [0., 1., 0., 0., 0.],
>           [0., 0., 1., 0., 0.],
>           [0., 0., 0., 1., 0.],
>           [0., 0., 0., 0., 1.]])



#### 6.5 np.random模块创建


```python
# 设置随机种子  作用是：使得每次随机的结果固定
np.random.seed(666)
```


```python
# 返回数据在[0,1)之间  包含0 不包含1
np.random.rand(5)
```




>    array([0.70043712, 0.84418664, 0.67651434, 0.72785806, 0.95145796])




```python
# 随机生成 2行5列的2维数组  数据在[0,1)之间
np.random.rand(2,5)
```




>    array([[0.0127032 , 0.4135877 , 0.04881279, 0.09992856, 0.50806631],
>           [0.20024754, 0.74415417, 0.192892  , 0.70084475, 0.29322811]])




```python
# 生成随机整数  区间范围：[2,5) 左闭右开  包含2不包含5
np.random.randint(2,5)
```




>    2




```python
# 指定shape
np.random.randint(2,5,5)
```




>    array([2, 2, 2, 2, 2])




```python
#uniform() 在 [low,high) 之间 生成均匀分布的数字
np.random.uniform(2,5,10)
```




>    array([4.30223703, 4.85969412, 2.87292148, 4.54334592, 3.04928571,
>           4.77169077, 2.88468359, 3.57314184, 4.82761689, 2.22421847])




```python
# 前面的 2， 5 表示 取值的区间范围 [2,5)
# 后面的(2,5) 是指定的shape
np.random.uniform(2,5,(2,5))
```




>    array([[2.82938752, 3.40275649, 2.94744596, 3.17048776, 2.80498943],
>           [4.26099152, 4.0002124 , 4.61863861, 3.56329157, 4.25061275]])




```python
# randn  返回数据具有标准正态分布 
# 即：均值为0 方差为1
np.random.randn(5)
```




>    array([-1.21715355, -0.99494737, -1.56448586, -1.62879004,  1.23174866])




```python
np.random.randn(2,5)
```




>    array([[-0.91360034, -0.27084407,  1.42024914, -0.98226439,  0.80976498],
>           [ 1.85205227,  1.67819021, -0.98076924,  0.47031082,  0.18226991]])




```python
# normal()  可以指定均值和标准差差
np.random.normal(1,10,5)
```




>    array([-7.43882492,  3.0996833 ,  3.29586662,  3.63076422, 22.66332222])




```python
np.random.normal(1,10,(2,5))
```




>    array([[ -9.48875925, -17.47684423,   6.34015028, -10.95748024,
>             -1.89157372],
>           [ -1.43327034,  -6.42666527,  13.07905106,  11.27877114,
>              1.57388197]])




```python
# choice 从给定的数组里 生成随机结果
np.random.choice(5, 3)
# 等同于 np.random.choice(0, 5, 3)  在[0,5)的区间范围内 生成3个数据
```




>    array([0, 1, 2])




```python
np.random.choice(5, (2, 3))
```




>    array([[0, 1, 0],
>           [0, 1, 2]])




```python
np.random.choice([1,3,5,2,3], 3)
```




>    array([3, 3, 5])




```python
# shuffle 把一个数组进行随机排列
a = np.arange(10)
np.random.shuffle(a) 
a
```




>    array([6, 3, 5, 1, 0, 8, 4, 9, 2, 7])




```python
a = np.arange(20).reshape(4, 5)
a
```




>    array([[ 0,  1,  2,  3,  4],
>           [ 5,  6,  7,  8,  9],
>           [10, 11, 12, 13, 14],
>           [15, 16, 17, 18, 19]])




```python
# 如果数组是多维的  则只会在第一维度打散数据
np.random.shuffle(a)
a
```




>    array([[10, 11, 12, 13, 14],
>           [15, 16, 17, 18, 19],
>           [ 5,  6,  7,  8,  9],
>           [ 0,  1,  2,  3,  4]])




```python
# permutation 把一个数组进行随机排列  或者数字的全排列
np.random.permutation(10)
```




>    array([4, 9, 8, 7, 3, 5, 6, 1, 0, 2])




```python
arr = np.arange(9).reshape((3,3))
arr
```




>    array([[0, 1, 2],
>           [3, 4, 5],
>           [6, 7, 8]])




```python
# permutation 与 shuffle 函数功能相同  区别在于：
# 注意 permutation不会更改原来的arr 会返回一个新的copy
np.random.permutation(arr)
```




>    array([[3, 4, 5],
>           [6, 7, 8],
>           [0, 1, 2]])




```python
arr
```




>    array([[0, 1, 2],
>           [3, 4, 5],
>           [6, 7, 8]])



 ### 7. numpy的数组索引
 三种索引方法：
 - 基础索引
 - 神奇索引
 - 布尔索引

#### 7.1 基础索引


```python
# 一维向量
x = np.arange(10)
x
```




>    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 二维向量 一般用大写字母
Y = np.arange(20).reshape(4, 5)
Y
```




>    array([[ 0,  1,  2,  3,  4],
>           [ 5,  6,  7,  8,  9],
>           [10, 11, 12, 13, 14],
>           [15, 16, 17, 18, 19]])




```python
# 取索引从2 到 倒数第一 个（不包含倒数第一个）
x[2:-1]
```




>    array([2, 3, 4, 5, 6, 7, 8])




```python
# 取 第1行第1列的数 
# 注意： 索引是从0开始的    我们日常所说的第1行 它的索引值是0
Y[0, 0]
```




>    0




```python
# 取索引为第2行的数据
Y[2]
```




>    array([10, 11, 12, 13, 14])




```python
# 取索引为第2列的数据
Y[:,2]
```




>    array([ 2,  7, 12, 17])



**注意：切片的修改会修改原来的数组**

原因： numpy经常要处理大数组，避免每次都复制


```python
# 修改x的第0-2（不包含2）数据
x[:2] = 666
x
```




>    array([666, 666,   2,   3,   4,   5,   6,   7,   8,   9])



#### 7.2 神奇索引
就是： 用整数数组进行的索引


```python
x = np.arange(10)
x
```




>    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
x[[2,3,4]]
```




>    array([2, 3, 4])




```python
indexs = np.array([[0,2],[1,3]])
x[indexs]
```




>    array([[0, 2],
>           [1, 3]])



**实例 ： 获取数组中最大的前n个数字**


```python
# 随机生成1-100之间的10个数字
arr = np.random.randint(1, 100, 10)
arr
```




>    array([37, 30, 76, 20, 63, 80, 42, 83, 91, 67])




```python
# arr.argsort() 会返回排序后的索引index
# 取最大值对应的3个下标
arr.argsort()[-3:]
```




>    array([5, 7, 8], dtype=int64)




```python
arr[arr.argsort()[-3:]]
```




>    array([80, 83, 91])



**二维数组**


```python
Y = np.arange(20).reshape(4, 5)
Y
```




>    array([[ 0,  1,  2,  3,  4],
>           [ 5,  6,  7,  8,  9],
>           [10, 11, 12, 13, 14],
>           [15, 16, 17, 18, 19]])




```python
# 筛选多行 列可以省略
Y[[0,2]]
```




>    array([[ 0,  1,  2,  3,  4],
>           [10, 11, 12, 13, 14]])




```python
Y[[0,2],:]
```




>    array([[ 0,  1,  2,  3,  4],
>           [10, 11, 12, 13, 14]])




```python
# 筛选多列 行不能省略
Y[:,[0, 2]]
```




>    array([[ 0,  2],
>           [ 5,  7],
>           [10, 12],
>           [15, 17]])




```python
# 同时指定行列
Y[[0,2,3],[1,2,3]]
```




>    array([ 1, 12, 18])



#### 7.3布尔索引


```python
x = np.arange(10)
x
```




>    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 返回bool值的数组
x>5
```




>    array([False, False, False, False, False, False,  True,  True,  True,
>            True])




```python
x[x>5]
```




>    array([6, 7, 8, 9])




```python
# 通过条件进行赋值
x[x<=5] = 0
x[x>5] = 1
x
```




>    array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])




```python
x = np.arange(10)
x[x<5] += 20
x
```




>    array([20, 21, 22, 23, 24,  5,  6,  7,  8,  9])




```python
Y = np.arange(20).reshape(4, 5)
Y
```




>    array([[ 0,  1,  2,  3,  4],
>           [ 5,  6,  7,  8,  9],
>           [10, 11, 12, 13, 14],
>           [15, 16, 17, 18, 19]])




```python
Y > 5
```




>    array([[False, False, False, False, False],
>           [False,  True,  True,  True,  True],
>           [ True,  True,  True,  True,  True],
>           [ True,  True,  True,  True,  True]])




```python
# Y>5的boolean数组 既有行又有列 因此返回的是 行列一维数组
Y[Y>5]
```




>    array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])




```python
Y[:, 3]
```




>    array([ 3,  8, 13, 18])




```python
Y[:, 3]>5
```




>    array([False,  True,  True,  True])




```python
# 把第3列大于5的行数据筛选出来
Y[:, 3][Y[:, 3]>5]
```




>    array([ 8, 13, 18])



**条件的组合**


```python
x = np.arange(10)
x
```




>    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
condition = (x%2==0)| (x>7)
condition
```




>    array([ True, False,  True, False,  True, False,  True, False,  True,
>            True])




```python
x[condition]
```




>    array([0, 2, 4, 6, 8, 9])



### 8. numpy的操作与函数


```python
A  = np.arange(6).reshape(2, 3)
A
```




>    array([[0, 1, 2],
>           [3, 4, 5]])




```python
# 相当于 A中的每个数据都+1
A+1
```




>    array([[1, 2, 3],
>           [4, 5, 6]])




```python
# 相当于 A中的每个数据都*3
A*3
```




>    array([[ 0,  3,  6],
>           [ 9, 12, 15]])




```python
np.sin(A)
```




>    array([[ 0.        ,  0.84147098,  0.90929743],
>           [ 0.14112001, -0.7568025 , -0.95892427]])




```python
np.exp(A)
```




>    array([[  1.        ,   2.71828183,   7.3890561 ],
>           [ 20.08553692,  54.59815003, 148.4131591 ]])




```python
B  = np.arange(6,12).reshape(2, 3)
B
```




>    array([[ 6,  7,  8],
>           [ 9, 10, 11]])




```python
# 对应位置元素相加
A + B
```




>    array([[ 6,  8, 10],
>           [12, 14, 16]])




```python
# 对应位置元素相减
A - B 
```




>    array([[-6, -6, -6],
>           [-6, -6, -6]])




```python
# 对应位置元素相乘
A*B
```




>    array([[ 0,  7, 16],
>           [27, 40, 55]])



**numpy的数学统计函数**


```python
arr = np.arange(12).reshape(3,4)
arr
```




>    array([[ 0,  1,  2,  3],
>           [ 4,  5,  6,  7],
>           [ 8,  9, 10, 11]])




```python
# 求和
np.sum(arr)
```




>    66




```python
# 乘积
np.prod(arr)
```




>    0




```python
# 累加
np.cumsum(arr)
```




>    array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 66], dtype=int32)




```python
# 累乘
np.cumprod(arr)
```




>    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)




```python
# 最小值
np.min(arr)
```




>    0




```python
# 最大值
np.max(arr)
```




>    11




```python
# 求取数列第?分位的数值
np.percentile(arr,[25,50,75])
```




>    array([2.75, 5.5 , 8.25])




```python
# 功能同上面  只不过范围为0-1直接
np.quantile(arr,[0.25,0.5,0.75])
```




>    array([2.75, 5.5 , 8.25])




```python
#中位数
np.median(arr)
```




>    5.5




```python
# 平均值
np.mean(arr)
```




>    5.5




```python
# 标准差
np.std(arr)
```




>    3.452052529534663




```python
# 方差
np.var(arr)
```




>    11.916666666666666




```python
# 加权平均
# weights 的 shape 需要和 arr 一样
weights = np.random.rand(*arr.shape)
np.average(arr, weights=weights)
```




>    5.355698948848374



### 9.numpy的axis参数

axis=0代表行  axis=1 代表列

对于sum/mean/media等聚合函数：
- axis=0 代表把行消解掉，axis=1 代表把列消解掉
- axis=0 代表跨行计算， axis=1 代表跨列计算


```python
arr = np.arange(12).reshape(3,4)
arr
```




>    array([[ 0,  1,  2,  3],
>           [ 4,  5,  6,  7],
>           [ 8,  9, 10, 11]])




```python
arr.sum(axis=0)
```




>    array([12, 15, 18, 21])




```python
arr.sum(axis=1)
```




>    array([ 6, 22, 38])




```python
arr.cumsum(axis=0)
```




>    array([[ 0,  1,  2,  3],
>           [ 4,  6,  8, 10],
>           [12, 15, 18, 21]], dtype=int32)




```python
arr.cumsum(axis=1)
```




>    array([[ 0,  1,  3,  6],
>           [ 4,  9, 15, 22],
>           [ 8, 17, 27, 38]], dtype=int32)




```python
mean = np.mean(arr, axis=0)
mean
```




>    array([4., 5., 6., 7.])




```python
std = np.std(arr,axis=0)
std
```




>    array([3.26598632, 3.26598632, 3.26598632, 3.26598632])




```python
result = arr-mean
result
```




>    array([[-4., -4., -4., -4.],
>           [ 0.,  0.,  0.,  0.],
>           [ 4.,  4.,  4.,  4.]])



### 10.numpy给数组添加维度


```python
arr = np.arange(5)
arr
```




>    array([0, 1, 2, 3, 4])




```python
arr.shape
```




>    (5,)



**方法1 np.newaxis关键字**

np.newaxis 就是None的别名


```python
np.newaxis is None
```




>    True




```python
np.newaxis == None
```




>    True




```python
arr[np.newaxis, :]
```




>    array([[0, 1, 2, 3, 4]])




```python
arr[np.newaxis, :].shape
```




>    (1, 5)




```python
# 给一维向量添加一个列维度
arr[:, np.newaxis]
```




>    array([[0],
>           [1],
>           [2],
>           [3],
>           [4]])




```python
arr[:,np.newaxis].shape
```




>    (5, 1)



**方法2 np.expand_dims 方法**


```python
np.expand_dims(arr, axis=0)
```




>    array([[0, 1, 2, 3, 4]])




```python
np.expand_dims(arr,axis=0).shape
```




>    (1, 5)




```python
np.expand_dims(arr,axis=1)
```




>    array([[0],
>           [1],
>           [2],
>           [3],
>           [4]])




```python
np.expand_dims(arr,axis=1).shape
```




>    (5, 1)



**方法3 np.reshape**


```python
np.reshape(arr, (1,5))
```




>    array([[0, 1, 2, 3, 4]])




```python
# -1表示 自动计算出结果  
np.reshape(arr,(1,-1))
```




>    array([[0, 1, 2, 3, 4]])




```python
np.reshape(arr,(1,-1)).shape
```




>    (1, 5)




```python
np.reshape(arr,(-1,1))
```




>    array([[0],
>           [1],
>           [2],
>           [3],
>           [4]])




```python
np.reshape(arr,(-1,1)).shape
```




>    (5, 1)



### 11.数组合并操作

**合并行**


```python
a = np.arange(6).reshape(2,3)
b = np.arange(6,18).reshape(4,3)
```


```python
a
```




>    array([[0, 1, 2],
>           [3, 4, 5]])




```python
b
```




>    array([[ 6,  7,  8],
>           [ 9, 10, 11],
>           [12, 13, 14],
>           [15, 16, 17]])




```python
np.concatenate([a,b])
```




>    array([[ 0,  1,  2],
>           [ 3,  4,  5],
>           [ 6,  7,  8],
>           [ 9, 10, 11],
>           [12, 13, 14],
>           [15, 16, 17]])




```python
np.vstack([a,b])
```




>    array([[ 0,  1,  2],
>           [ 3,  4,  5],
>           [ 6,  7,  8],
>           [ 9, 10, 11],
>           [12, 13, 14],
>           [15, 16, 17]])




```python
np.row_stack([a,b])
```




>    array([[ 0,  1,  2],
>           [ 3,  4,  5],
>           [ 6,  7,  8],
>           [ 9, 10, 11],
>           [12, 13, 14],
>           [15, 16, 17]])



**合并列**


```python
a = np.arange(12).reshape(3,4)
a
```




>    array([[ 0,  1,  2,  3],
>           [ 4,  5,  6,  7],
>           [ 8,  9, 10, 11]])




```python
b = np.arange(12,18).reshape(3,2)
b
```




>    array([[12, 13],
>           [14, 15],
>           [16, 17]])




```python
np.concatenate([a,b],axis=1)
```




>    array([[ 0,  1,  2,  3, 12, 13],
>           [ 4,  5,  6,  7, 14, 15],
>           [ 8,  9, 10, 11, 16, 17]])




```python
np.hstack([a,b])
```




>    array([[ 0,  1,  2,  3, 12, 13],
>           [ 4,  5,  6,  7, 14, 15],
>           [ 8,  9, 10, 11, 16, 17]])




```python
np.column_stack([a,b])
```




>    array([[ 0,  1,  2,  3, 12, 13],
>           [ 4,  5,  6,  7, 14, 15],
>           [ 8,  9, 10, 11, 16, 17]])



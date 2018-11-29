# Numpy快速入门(Numpy 1.14 官方文档中文翻译)

本教程是基于Numpy1.14官方网站的文档

原文地址：[点我呀](https://docs.scipy.org/doc/numpy/user/quickstart.html#quickstart-tutorial)



## 1. 准备工作

在你开始阅读本教程之前，你需要了解一些Python知识。如果你想复习一下，请[点我](http://docs.python.org/tut/)查看Python官网教程。

如果你想运行一下本教程的相关例子，你需要在你的电脑上安装一些软件，请[点我](http://scipy.org/install.html)查看相关安装说明。

## ２.基础知识

Numpy的主要操作对象是同类的多维数组，即一个由相同类型元素（通常是数字）组成的、以正数为索引的数据表。在Numpy里面，维度称为“轴”。

举例来说，三维空间内一点的坐标`[1,2,1]`有一个轴，三个元素，所以我们通常称它的长度为3。在以下所示的例子中，数组有两个轴，第一个轴的长度为2，第二个轴的长度为3。

```python
[[ 1., 0., 0.],
 [ 0., 1., 2.]]
```

Numpy的数组类型叫做`ndarray`，也就是Numpy数组(以下简称为数组)。需要注意的是,`numpy.array`不同于Python标准库中的`array.array`，后者只处理一维的数组并且提供了很少的功能。一个`ndarray`对象有以下一些重要的属性：

- **ndarray.ndim**

  数组的轴的数量，即维度数量。

- **ndarray.shape**

  数组的维度。返回的是一个整数元组，指示了一个数组在各个维度的大小。对于一个n行m列的矩阵来说，它的`shape`是`(n,m)`。`shape`的元组长度因此是轴的数量，即`ndim`。

- **ndarray.size**

  数组所有元素的数量，等于`shape`返回元组元素的乘积。

- **ndarray.dtype**

  一个用于描述数组元素类型的对象。可以用标准Python类型来创造或指定dtype的类型。另外，Numpy也提供了自己的类型，如`numpy.int32`，`numpy.int16`，`numpy.float64`等。

- **ndarray.itemsize**

  数组每个元素的字节大小。比如一个数组的元素为`float64`，它的`itemsize`为8(=64/8)，

  `complex32`的`itemsize`为4(=32/8)。这个属性等同于`ndarray.dtype.itemsize`。

- **ndarray.data**

  包含了数组每个实际元素的缓冲器。一般来说我们不会用到这个属性因为我们可以通过索引工具来获取到数组的每个元素的值。

  #### 一些具体的例子

  ------

  ```python
  >>> import numpy as np
  >>> a = np.arange(15).reshape(3, 5)
  >>> a
  array([[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14]])
  >>> a.shape
  (3, 5)
  >>> a.ndim
  2
  >>> a.dtype.name
  'int64'
  >>> a.itemsize
  8
  >>> a.size
  15
  >>> type(a)
  <type 'numpy.ndarray'>
  >>> b = np.array([6, 7, 8])
  >>> b
  array([6, 7, 8])
  >>> type(b)
  <type 'numpy.ndarray'>
  ```

  #### 创建数组

  ------

  有很多种创建数组的方法。

  比如你可以通过`array`函数将一个标准的Python列表或者元组转换为Numpy数组。最终得到的数组的元素类型将从被转换的序列元素推算得到。

  ```python
  >>> import numpy as np
  >>> a = np.array([2,3,4])
  >>> a
  array([2, 3, 4])
  >>> a.dtype
  dtype('int64')
  >>> b = np.array([1.2, 3.5, 5.1])
  >>> b.dtype
  dtype('float64')
  ```

  一个常见的调用`array`函数的错误是提供了由多个数字组成的参数表，而不是单个的列表。

  ```python
  >>> a = np.array(1,2,3,4)    # WRONG
  >>> a = np.array([1,2,3,4])  # RIGHT
  ```

  如果待转换的序列中包含子序列，`array`函数将会将其转换为二维数组，如果子序列还包含子序列，就会被转换为三维数组，依次类推。

  ```python
  >>> b = np.array([(1.5,2,3), (4,5,6)])
  >>> b
  array([[ 1.5,  2. ,  3. ],
         [ 4. ,  5. ,  6. ]])
  ```

  数组的元素类型也可以在创建时进行明确指定。

  ```python
  >>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
  >>> c
  array([[ 1.+0.j,  2.+0.j],
         [ 3.+0.j,  4.+0.j]])
  ```

  在很多情况下，数组元素的类型时不明确的，但它的大小时明确的。因此，Numpy提供了几个函数用于创建包含占位内容的数组。这使得填充数组这一非常消耗资源的操作得以最简化。

  `zeros`函数可以创造一个全0的数组，`ones`可以创建一个全1的数组，`empty`创建一个随机内容的数组。所创建数组的默认类型是`float64`。

  ```python
  >>> np.zeros( (3,4) )
  array([[ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.]])
  >>> np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
  array([[[ 1, 1, 1, 1],
          [ 1, 1, 1, 1],
          [ 1, 1, 1, 1]],
         [[ 1, 1, 1, 1],
          [ 1, 1, 1, 1],
          [ 1, 1, 1, 1]]], dtype=int16)
  >>> np.empty( (2,3) )                                 # uninitialized, output may vary
  array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
         [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
  ```

  为创建一个数字序列数组，Numpy提供了类似于`range`的函数，可以返回数组而不是列表。

  ```python
  >>> np.arange( 10, 30, 5 )
  array([10, 15, 20, 25])
  >>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
  array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
  ```

  当`arange`的参数是浮点数时，通常很难预测最终数组所包含的数字元素，原因是Python的浮点数精度有限。正是因为这个原因，通常更好的选择是使用`linspace`函数接收一个我们做需要的元素个数的参数，而不是步长。

  ```python
  >>> from numpy import pi
  >>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
  array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
  >>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
  >>> f = np.sin(x)
  ```

  Numpy还有以下函数，详情请点击具体的函数名称：

  - [array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html#numpy.array)
  - [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html#numpy.zeros)
  - [zeros_like](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html#numpy.zeros_like)
  - [ones](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.ones)
  - [ones_like](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones_like.html#numpy.ones_like)
  - [empty](https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html#numpy.empty)
  - [empty_like](https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty_like.html#numpy.empty_like)
  - [arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange)
  - [linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html#numpy.linspace)
  - [numpy.random.rand](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html#numpy.random.rand)
  - [numpy.random.randn](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn)
  - [fromfunction](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfunction.html#numpy.fromfunction)
  - [fromfile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfile.html#numpy.fromfile)

  #### 打印数组

  ------

  打印数组时，Numpy会像打印层层嵌套的列表一样，但是会有以下的布局：

  - 最后的轴会是从左向右打印
  - 倒数第二个轴式自顶向下打印
  - 剩余的也是自顶向下打印，每一片与下一片的之间会有一个空行

  一维数组会按排打印，二维数组按矩阵打印，三维数组按矩阵列表打印。

  ```python
  >>> a = np.arange(6)                         # 1d array
  >>> print(a)
  [0 1 2 3 4 5]
  >>>
  >>> b = np.arange(12).reshape(4,3)           # 2d array
  >>> print(b)
  [[ 0  1  2]
   [ 3  4  5]
   [ 6  7  8]
   [ 9 10 11]]
  >>>
  >>> c = np.arange(24).reshape(2,3,4)         # 3d array
  >>> print(c)
  [[[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [16 17 18 19]
    [20 21 22 23]]]
  ```

  关于`reshape`函数，将会在后文详细介绍。

  如果一个数组太大了无法打印，Numpy会自动跳过中间部分的元素只打印边界元素。

  ```python
  >>> print(np.arange(10000))
  [   0    1    2 ..., 9997 9998 9999]
  >>>
  >>> print(np.arange(10000).reshape(100,100))
  [[   0    1    2 ...,   97   98   99]
   [ 100  101  102 ...,  197  198  199]
   [ 200  201  202 ...,  297  298  299]
   ...,
   [9700 9701 9702 ..., 9797 9798 9799]
   [9800 9801 9802 ..., 9897 9898 9899]
   [9900 9901 9902 ..., 9997 9998 9999]]
  ```

  若要禁止这种行为强行使Numpy打印出整个数组，你可以通过`set_printoptions`来改变打印方式。

  ```python
  >>> np.set_printoptions(threshold=np.nan)
  ```

  #### 基础操作

  ------

  算数运算符是对每个元素进行运算的，运算结果会放入一个新的数组中返回。

  ```python
  >>> a = np.array( [20,30,40,50] )
  >>> b = np.arange( 4 )
  >>> b
  array([0, 1, 2, 3])
  >>> c = a-b
  >>> c
  array([20, 29, 38, 47])
  >>> b**2
  array([0, 1, 4, 9])
  >>> 10*np.sin(a)
  array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
  >>> a<35
  array([ True, True, False, False])
  ```

  与一些进行矩阵运算的语言不同的是，乘积运算符`*`是矩阵相应位置上的元素进行乘运算的。而矩阵的乘法运算可以使用`dot`方法。

  ```python
  >>> A = np.array( [[1,1],
  ...             [0,1]] )
  >>> B = np.array( [[2,0],
  ...             [3,4]] )
  >>> A*B                         # elementwise product
  array([[2, 0],
         [0, 4]])
  >>> A.dot(B)                    # matrix product
  array([[5, 4],
         [3, 4]])
  >>> np.dot(A, B)                # another matrix product
  array([[5, 4],
         [3, 4]])
  ```

  一些像`+=`和`*=`的运算符，结果是对原数组进行修改而不是创建一个新数组。

  ```python
  >>> a = np.ones((2,3), dtype=int)
  >>> b = np.random.random((2,3))
  >>> a *= 3
  >>> a
  array([[3, 3, 3],
         [3, 3, 3]])
  >>> b += a
  >>> b
  array([[ 3.417022  ,  3.72032449,  3.00011437],
         [ 3.30233257,  3.14675589,  3.09233859]])
  >>> a += b                  # b is not automatically converted to integer type
  Traceback (most recent call last):
    ...
  TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
  ```

  当对不同类型的数组进行操作时，结果的数组类型会是更普遍或者更精确的类型(一种称为精度提升的操作)。

  ```python
  >>> a = np.ones(3, dtype=np.int32)
  >>> b = np.linspace(0,pi,3)
  >>> b.dtype.name
  'float64'
  >>> c = a+b
  >>> c
  array([ 1.        ,  2.57079633,  4.14159265])
  >>> c.dtype.name
  'float64'
  >>> d = np.exp(c*1j)
  >>> d
  array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
         -0.54030231-0.84147098j])
  >>> d.dtype.name
  'complex128'
  ```

  一些一元操作符，比如计算数组所有元素的和，是对`ndarray`类型对象进行操作。

  ```python
  >>> a = np.random.random((2,3))
  >>> a
  array([[ 0.18626021,  0.34556073,  0.39676747],
         [ 0.53881673,  0.41919451,  0.6852195 ]])
  >>> a.sum()
  2.5718191614547998
  >>> a.min()
  0.1862602113776709
  >>> a.max()
  0.6852195003967595
  ```

  这些操作默认是把一个数组当作一个数字列表来进行，无论它的`shape`是怎样的。然而，指定`axis`参数可以在指定的轴上进行操作。

  ```python
  >>> b = np.arange(12).reshape(3,4)
  >>> b
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]])
  >>>
  >>> b.sum(axis=0)                            # sum of each column
  array([12, 15, 18, 21])
  >>>
  >>> b.min(axis=1)                            # min of each row
  array([0, 4, 8])
  >>>
  >>> b.cumsum(axis=1)                         # cumulative sum along each row
  array([[ 0,  1,  3,  6],
         [ 4,  9, 15, 22],
         [ 8, 17, 27, 38]])
  ```

  #### 通用函数

  ------

  Numpy提供了一些我们所熟知的数学函数如sin,cos和exp。在Numpy中，这些函数被称为通用函数(`ufunc`)。在Numpy中，这些函数依次操作于每一个元素上，最终返回一个用于存放结果的新数组。

  ```python
  >>> B = np.arange(3)
  >>> B
  array([0, 1, 2])
  >>> np.exp(B)
  array([ 1.        ,  2.71828183,  7.3890561 ])
  >>> np.sqrt(B)
  array([ 0.        ,  1.        ,  1.41421356])
  >>> C = np.array([2., -1., 4.])
  >>> np.add(B, C)
  array([ 2.,  0.,  6.])
  ```

  Numpy还有以下函数，详情请点击具体的函数名称：

  - [all](https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html#numpy.all)
  - [any](https://docs.scipy.org/doc/numpy/reference/generated/numpy.any.html#numpy.any)
  - [apply_along_axis](https://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis)
  - [argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html#numpy.argmax)
  - [argmin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html#numpy.argmin)
  - [argsort](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html#numpy.argsort)
  - [average](https://docs.scipy.org/doc/numpy/reference/generated/numpy.average.html#numpy.average)
  - [bincount](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html#numpy.bincount)
  - [ceil](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ceil.html#numpy.ceil)
  - [clip](https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html#numpy.clip)
  - [conj](https://docs.scipy.org/doc/numpy/reference/generated/numpy.conj.html#numpy.conj)
  - [corrcoef](https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html#numpy.corrcoef)
  - [cov](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html#numpy.cov)
  - [cross](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cross.html#numpy.cross)
  - [cumprod](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumprod.html#numpy.cumprod)
  - [cumsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html#numpy.cumsum)
  - [diff](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diff.html#numpy.diff)
  - [dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot)
  - [floor](https://docs.scipy.org/doc/numpy/reference/generated/numpy.floor.html#numpy.floor)
  - [inner](https://docs.scipy.org/doc/numpy/reference/generated/numpy.inner.html#numpy.inner)
  - [lexsort](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html#numpy.lexsort)
  - [max](https://docs.python.org/dev/library/functions.html#max)
  - [maximum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html#numpy.maximum)
  - [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean)
  - [median](https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html#numpy.median)
  - [min](https://docs.python.org/dev/library/functions.html#min)
  - [minium](https://docs.scipy.org/doc/numpy/reference/generated/numpy.minimum.html#numpy.minimum)
  - [nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html#numpy.nonzero)
  - [outer](https://docs.scipy.org/doc/numpy/reference/generated/numpy.outer.html#numpy.outer)
  - [prod](https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html#numpy.prod)
  - [re](https://docs.python.org/dev/library/re.html#module-re)
  - [round](https://docs.python.org/dev/library/functions.html#round)
  - [sort](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html#numpy.sort)
  - [std](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html#numpy.std)
  - [sum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html#numpy.sum)
  - [trace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.trace.html#numpy.trace)
  - [transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html#numpy.transpose)
  - [var](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html#numpy.var)
  - [vdot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vdot.html#numpy.vdot)
  - [vectorize](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html#numpy.vectorize)
  - [where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html#numpy.where)

  #### 索引，切片和迭代

  ------

  一维数组可以像数组和其他Python序列一样进行索引、切片和迭代。

  ```python
  >>> a = np.arange(10)**3
  >>> a
  array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
  >>> a[2]
  8
  >>> a[2:5]
  array([ 8, 27, 64])
  >>> a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
  >>> a
  array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
  >>> a[ : :-1]                                 # reversed a
  array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
  >>> for i in a:
  ...     print(i**(1/3.))
  ...
  nan
  1.0
  nan
  3.0
  nan
  5.0
  6.0
  7.0
  8.0
  9.0
  ```

  多维数组在每一个轴上都可以进行索引，索引的指数会在一个由逗号分隔的元组中给出。

  ```python
  >>> def f(x,y):
  ...     return 10*x+y
  ...
  >>> b = np.fromfunction(f,(5,4),dtype=int)
  >>> b
  array([[ 0,  1,  2,  3],
         [10, 11, 12, 13],
         [20, 21, 22, 23],
         [30, 31, 32, 33],
         [40, 41, 42, 43]])
  >>> b[2,3]
  23
  >>> b[0:5, 1]                       # each row in the second column of b
  array([ 1, 11, 21, 31, 41])
  >>> b[ : ,1]                        # equivalent to the previous example
  array([ 1, 11, 21, 31, 41])
  >>> b[1:3, : ]                      # each column in the second and third row of b
  array([[10, 11, 12, 13],
         [20, 21, 22, 23]])
  ```

  当给出的索引指数比轴的数目更少时，缺的指数可以考虑用`:`补全。

  ```python
  >>> b[-1]                                  # the last row. Equivalent to b[-1,:]
  array([40, 41, 42, 43])
  ```

  `b[i]`括号中的表达式被视为一个`i`，后跟`:`，在需要的时候代表剩余的轴。Numpy也允许这样写`b[i,...]`。

  `...`这样的点代表了要生成一个完整索引元组所需要数量的栏的数目。比如若`x`是一个有五个轴的数组，那么：

  -  `x[1,2,...]` 等同于 `x[1,2,:,:,:]`，
  -  `x[...,3]` 等同于 `x[:,:,:,:,3]`  ，
  -  `x[4,...,5,:]`等同于`x[4,:,:,5,:]`。

  ```python
  >>> c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
  ...                 [ 10, 12, 13]],
  ...                [[100,101,102],
  ...                 [110,112,113]]])
  >>> c.shape
  (2, 2, 3)
  >>> c[1,...]                                   # same as c[1,:,:] or c[1]
  array([[100, 101, 102],
         [110, 112, 113]])
  >>> c[...,2]                                   # same as c[:,:,2]
  array([[  2,  13],
         [102, 113]])
  ```

  多维数组的迭代在操作时相当于关于第一个轴的迭代：

  ```python
  >>> for row in b:
  ...     print(row)
  ...
  [0 1 2 3]
  [10 11 12 13]
  [20 21 22 23]
  [30 31 32 33]
  [40 41 42 43]
  ```

  然而，如果要在数组每一个元素上进行操作，可以使用`flat`属性，这是一个数组所有元素的迭代器：

  ```python
  >>> for element in b.flat:
  ...     print(element)
  ...
  0
  1
  2
  3
  10
  11
  12
  13
  20
  21
  22
  23
  30
  31
  32
  33
  40
  41
  42
  43
  ```

  ## 3.Shape操作

  #### 改变数组的shape

  一个数组的shape在创建时由每个轴的元素数量决定。

  ```python
  >>> a = np.floor(10*np.random.random((3,4)))
  >>> a
  array([[ 2.,  8.,  0.,  6.],
         [ 4.,  5.,  1.,  1.],
         [ 8.,  9.,  3.,  6.]])
  >>> a.shape
  (3, 4)
  ```

  数组的shape可以通过各种命令进行修改。需要注意的是，通过以下三种命令修改shape会返回一个经过修改的数组，而不会改变原数组的元素：

  ```python
  >>> a.ravel()  # returns the array, flattened
  array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
  >>> a.reshape(6,2)  # returns the array with a modified shape
  array([[ 2.,  8.],
         [ 0.,  6.],
         [ 4.,  5.],
         [ 1.,  1.],
         [ 8.,  9.],
         [ 3.,  6.]])
  >>> a.T  # returns the array, transposed
  array([[ 2.,  4.,  8.],
         [ 8.,  5.,  9.],
         [ 0.,  1.,  3.],
         [ 6.,  1.,  6.]])
  >>> a.T.shape
  (4, 3)
  >>> a.shape
  (3, 4)
  ```

  由`reval()`函数返回的数组元素的顺序是典型的C语言风格，最右边的索引是改变最快的，因此`a[0,0]`之后的元素是`a[0,1]`。如果数组的shape被改变，数组还是C语言风格的。Numpy会自然地创建存储在这种顺序的数组，所以`reval()`函数通常不需要复制它的参数，但是如果一个数组是通过从别的数组切片创建的，或者通过不寻常的操作创建的，参数可能需要复制。`reval()`函数和`reshape()`函数可以通过使用一个参数来使用FORTRAN风格的数组，在这种数组中，最左侧的索引是改变最快的。

  `reshape()`函数返回的是一个由参数决定shape的改变后的数组，而`ndarray.resize`则是修改数组本身：

  ```python
  >>> a
  array([[ 2.,  8.,  0.,  6.],
         [ 4.,  5.,  1.,  1.],
         [ 8.,  9.,  3.,  6.]])
  >>> a.resize((2,6))
  >>> a
  array([[ 2.,  8.,  0.,  6.,  4.,  5.],
         [ 1.,  1.,  8.,  9.,  3.,  6.]])
  ```

  如果在改变形状的操作中某维度的参数是-1，那么另外的维度会自动计算：

  ```python
  >>> a.reshape(3,-1)
  array([[ 2.,  8.,  0.,  6.],
         [ 4.,  5.,  1.,  1.],
         [ 8.,  9.,  3.,  6.]])
  ```

  #### 把不同的数组整合成一个数组

  几个不同的数组可以沿不同的轴整合在一起：

  ```python
  >>> a = np.floor(10*np.random.random((2,2)))
  >>> a
  array([[ 8.,  8.],
         [ 0.,  0.]])
  >>> b = np.floor(10*np.random.random((2,2)))
  >>> b
  array([[ 1.,  8.],
         [ 0.,  4.]])
  >>> np.vstack((a,b))
  array([[ 8.,  8.],
         [ 0.,  0.],
         [ 1.,  8.],
         [ 0.,  4.]])
  >>> np.hstack((a,b))
  array([[ 8.,  8.,  1.,  8.],
         [ 0.,  0.,  0.,  4.]])
  ```

  [column_stack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html#numpy.column_stack)把一个一维数组作为已栏整合进二维数组，相当于二维数组的[hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack)函数。

  ```python
  >>> from numpy import newaxis
  >>> np.column_stack((a,b))     # with 2D arrays
  array([[ 8.,  8.,  1.,  8.],
         [ 0.,  0.,  0.,  4.]])
  >>> a = np.array([4.,2.])
  >>> b = np.array([3.,8.])
  >>> np.column_stack((a,b))     # returns a 2D array
  array([[ 4., 3.],
         [ 2., 8.]])
  >>> np.hstack((a,b))           # the result is different
  array([ 4., 2., 3., 8.])
  >>> a[:,newaxis]               # this allows to have a 2D columns vector
  array([[ 4.],
         [ 2.]])
  >>> np.column_stack((a[:,newaxis],b[:,newaxis]))
  array([[ 4.,  3.],
         [ 2.,  8.]])
  >>> np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
  array([[ 4.,  3.],
         [ 2.,  8.]])
  ```

  另一方面，对于输入的数组`row_stack`函数相当于[vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)函数。一般来说，对于维数超过二的数组，[hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack)函数会沿着第二个轴进行整合，[vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)会沿着第一个轴进行整合，而[concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html#numpy.concatenate)函数会接受一个参数，并且沿着这个参数指定的轴进行整合。

  ##### Note

  在一些复杂的例子里面，当沿着某个轴堆叠数字时，[r_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html#numpy.r_)函数和[c_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html#numpy.c_)函数是非常有用的，并且可以使用范围量`:`。

  ```python
  >>> np.r_[1:4,0,4]
  array([1, 2, 3, 0, 4])
  ```

  当使用数组作为参数时，[r_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html#numpy.r_)函数和[c_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html#numpy.c_)函数与[vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)函数、[hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack)函数在行为上是很相似的，但是允许传入一个参数来决定哪个轴来整合。

  #### 将一个数组拆成若干个数组

  通过指定返回同型数组的数量或者指定在某栏之处进行拆分，[hsplit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hsplit.html#numpy.hsplit)函数可以将一个数组沿着水平的轴进行拆分：

  ```python
  >>> a = np.floor(10*np.random.random((2,12)))
  >>> a
  array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
         [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
  >>> np.hsplit(a,3)   # Split a into 3
  [array([[ 9.,  5.,  6.,  3.],
         [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
         [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
         [ 2.,  2.,  4.,  0.]])]
  >>> np.hsplit(a,(3,4))   # Split a after the third and the fourth column
  [array([[ 9.,  5.,  6.],
         [ 1.,  4.,  9.]]), array([[ 3.],
         [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
         [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
  ```

  [vsplit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vsplit.html#numpy.vsplit)可以沿着垂直的轴进行拆分，而[array_split](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html#numpy.array_split)允许指定沿着哪个轴进行拆分。

  ## 4.复制和视图

  当对数组进行操作时，其数据有时会被复制到一个新的数组里但有时不会。这可能会使入门者一脸懵逼，看看以下几个例子：

  #### 完全不复制

  一些像下面这种简单的操作不会进行数组对象或者数据的的复制：

  ```python
  >>> a = np.arange(12)
  >>> b = a            # no new object is created
  >>> b is a           # a and b are two names for the same ndarray object
  True
  >>> b.shape = 3,4    # changes the shape of a
  >>> a.shape
  (3, 4)
  ```

  Python将可变对象作为引用传递，所以函数调用不会进行复制：

  ```python
  >>> def f(x):
  ...     print(id(x))
  ...
  >>> id(a)                           # id is a unique identifier of an object
  148293216
  >>> f(a)
  148293216
  ```

  #### 视图或者浅复制

  不同的数组对象可以分享不同的数据，`view`函数会创建一个新数组对象，与原数组的数据相同。

  ```python
  >>> c = a.view()
  >>> c is a
  False
  >>> c.base is a                        # c is a view of the data owned by a
  True
  >>> c.flags.owndata
  False
  >>>
  >>> c.shape = 2,6                      # a's shape doesn't change
  >>> a.shape
  (3, 4)
  >>> c[0,4] = 1234                      # a's data changes
  >>> a
  array([[   0,    1,    2,    3],
         [1234,    5,    6,    7],
         [   8,    9,   10,   11]])
  ```

  切片操作会返回一个原数组的视图：

  ```python
  >>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
  >>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
  >>> a
  array([[   0,   10,   10,    3],
         [1234,   10,   10,    7],
         [   8,   10,   10,   11]])
  ```

  #### 深复制

  `copy`函数会对原数组对象和数据进行完全的复制：

  ```python
  >>> d = a.copy()                          # a new array object with new data is created
  >>> d is a
  False
  >>> d.base is a                           # d doesn't share anything with a
  False
  >>> d[0,0] = 9999
  >>> a
  array([[   0,   10,   10,    3],
         [1234,   10,   10,    7],
         [   8,   10,   10,   11]])
  ```

  #### 函数和方法一览

  以下会列出Numpy一些比较有用的函数和方法，你可以[点我](https://docs.scipy.org/doc/numpy/reference/routines.html#routines)来查看完整的列表。

  译者注：因为列表比较多，此处就不一一列举了，可以[点我前往](https://docs.scipy.org/doc/numpy/user/quickstart.html#functions-and-methods-overview)官网查看。
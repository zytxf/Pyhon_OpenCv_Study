**什么是局部程序块(local block)?**     

​          局部程序块是指一对大括号({})之间的一段C语言程序。一个C函数包含一对大括号，这对大括号之间的所有内容都包含在一个局部程序块中。if语句和swich语句也可以包含一对大括号，每对大括号之间的代码也属于一个局部程序块。此外，你完全可以创建你自己的局部程序块，而不使用C函数或基本的C语句。你可以在局部程序块中说明一些变量，这种变量被称为局部变量，它们只能在局部程序块的开始部分说明，并且只在说明它的局部程序块中有效。如果局部变量与局部程序块以外的变量重名，则前者优先于后者。

下面是一个使用局部程序块的例子：

```c
#include <stdio.h>

int main()
{
     /* Begin local block for function main() */
     int test_var = 10;
     printf("Test variable before the if statement: %d\n", test_var);
     if (test_var>5)
     {
           /* Begin local block for "if" statement */
           int test_var ;
           test_var = 5;
           printf("Test variable within the if statement: %d\n",test_var);
           {
                 /* Begin independent local block (not tied to any function or keyword) */
                  int test_var = 0;
                  printf ("Test variable within the independent local block: %d\n",test_var);
            }
      }
      /* End independent local block */
      printf ("Test variable after the if statement: %d\n", test_var);
      return 0;
} /*End local block for function main ()*/
```

上例产生如下输出结果：
Test variable before the if statement: 10
Test variable within the if statement: 5
Test variable within the independent local block:0
Test variable after the if statement: 10
​    注意，在这个例子中，每次test_var被定义时，它都要优先于前面所定义的test_var变量。此外还要注意，当if语句的局部程序块结束时，程序重新进入最初定义的test_var变量的作用范围，此时test_var的值为10。
​    可以把变量保存在局部程序块中吗?
​    用局部程序块来保存变量是不常见的，你应该尽量避免这样做，但也有极少数的例外。例如，为了调试程序，你可能要说明一个全局变量的局部实例，以便在相应的函数体内部进行测试。为了使程序的某一部分变得更易读，你也可能要使用局部程序块，例如，在接近变量被使用的地方说明一个变量有时就会使程序变得更易读。然而，编写得较好的程序通常不采用这种方式来说明变量，你应该尽量避免使用局部程序块来保存变量。
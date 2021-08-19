## C++



[toc]





### const用法

const本身是一个修饰符，被修饰的变量不可改变，具有只读的特点

1. const修饰普通变量: ```const int a = 10``` ，因为const变量不能被改变，所以必须定义的时候初始化。

2. const类对象，不能改变任何成员变量，以及不能调用任何非const成员函数。

3. const+指针

   ```
   const int* p; // 一个指针，保存const变量的地址，->不能通过该指针改变变量，可以改变指针的指向
   int* const p; // 一个不能被改变指向的指针
   ```

4. const 作为函数参数：```void func1(const string& a)``` 在该函数内变量不可更改

5. const成员函数，不能修改任何成员变量，除非成员变量用mutable修饰

   ```
   class A {
   public:
   	int a = 0;
   	int b = 0;
   	mutable int c = 0;
   	void func1(int a, int b, int c) const {
   		a = 1; // error
   		c = 1; // change c to 1
   	}
   }
   ```



#### 静态库和动态库

一般 ```.a``` 文件为静态库 ```.so``` 文件为动态库

目标文件(target file)：在Linux下为 Executable Linkable Format 格式，Win下为Portable Executable 格式

在使用外部库文件的时候，

静态库：编译时把用到的代码复制 -> 一个大的可执行文件， 只用到自己，兼容性差(原文件变动时要重新编译)

动态库：运行/加载时链接外部文件，文件大小比较小，兼容性好，加载速度略慢



### typedef

不是宏的字符串替代，定义一种类型的新的名字

1. 定义一个类型，eg: ```typedef char* PCHAR```

2. 在跨平台的时候，可以用typedef 来定义不同的类型

   ```
   platform1: typedef double D;
   platform2: typedef long double D;
   platform3: typedef float D;
   ```

3. 替换复杂声明

   ```
   original: void (*a)(void (*)());
   
   ->	typedef void(* fun1)();
   ->  typedef void(* fun2)(fun1);
   =>  fun1 a;
   ```



### C++文件流 ifstream

基本操作：

> ```ifstream::open(string <filename>, ifstream::[in(input) | out(output) | binary | ate(at end) | app(append) | trunc(truncate)] <mode>)```

> ```ifstream::close()```

> ```ifstream::rdbuf()```

当文件打开/关闭失败的时候，可以通过 ```state_flag``` 来判断

| STATE_FLAG | 含义                              |
| ---------- | --------------------------------- |
| goodbit    | No errors                         |
| eofbit     | end of file reached on input file |
| failbit    | Logic error on i/o operation      |
| Badbit     | Read/write error on i/o operation |

```  file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );```

可以用此来保证stream可以抛出异常



### inline

C++正常调用函数的时候，CPU会先取出函数对应的内存地址，把参数复制到堆栈然后跳转到函数的运行地址。函数运行完之后，把返回值存储到预定的内存地址，再跳转回原来的代码地址。因此调用函数的时候存在函数压/出栈的开销。

* ```inline``` 函数用以减少函数的跳转开销，调用时，直接替换函数的代码到内联函数的位置，在编译时执行。
* ```inline```不同于宏指令```#define```，内联函数会在编译时检查数据类型，而宏指令不会，只是字符串的替换。
* virtual 函数不能使用inline，因为虚函数是在运行时通过虚表确定函数的地址，而内联函数在编译时就需要替换代码。
* C++编译器可能会忽略特别长的内联函数，以减少代码容量。
* 一般inline要求写在头文件中，类的成员函数默认是inline，在类外部声明需要添加inline



感觉规定了inline写在类的内部以后就有点鸡肋了。毕竟在类的内部函数是默认为inline的，可以不用添加关键字，唯一的作用是在类外声明或者具体展开。不过本身编译器也会自行判断是否作为inline展开。。。



### 运算符重载

之前一直知道有这么个东西，但都没有了解太多。。。

C++中预定义的运算符只能用于基本数据类型，不能用于对象操作。又因为实际数据结构复杂了之后经常需要对象之间的操作，所以重载运算符可以使对象之间进行运算。

基本语法：

```
<return_type> operator <operator> (params)

eg:
Complex operator+ (const Complex& a, const Complex& b) {
		return Complex(a.real+b.real, a.img+b.img)
}
```

除了最常见的```+, - *, /```这类运算符重载，比较重要的是小括号```()```的重载

对于小括号```()```的重载，一般来说有两种用法，Callable和索引



Callable：

​		Callable 简单说就是类似于函数调用的时候 需要写成  ```Func1(params)``` 中```()``` 的作用。Callable，也就是可调用对象，包括了函数指针、重载operator()的对象以及可隐式转化为前两者的对象。

​		重载了```()``` 的对象被叫做 Functor， 即仿函数，大概就是说用法和长相和函数一样的对象吧。

​		函数传参的时候，可以使用函数指针来传入一个函数，同时也可以定义一个类，里面实现了一个函数。重载()意义在于简化这个类的功能，因为你定义了重载()的类之后可以像函数一样调用，用起来方便。相比于函数指针，仿函数是把它变成面向对象，多了一些状态变量。

​		用仿函数的另一个理由是C++的编译有优化，会把他展开成内联。



索引：

C++的```[]``` 运算符只能声明一个参数，所以很多矩阵、张量的操作可以通过重载```()```来近似的实现python中那种索引的效果。





### Lambda 表达式（匿名函数）

C++11 开始支持lambda表达式，即匿名函数。基本语法为：

```C++
[capture](parameters) -> return-type {body}  //普通形式，只有 [capture] 和 {body} 是必要的

[capture](parameters){body} //只有一条return 表达式或者 返回为void，此时->也可以去掉

/*
[capture]：capture声明了在lambda函数内部可以使用的外部变量，(引用/复制外部变量)

  []        //未定义变量.试图在Lambda内使用任何外部变量都是错误的.
  [x, &y]   //x 按值捕获, y 按引用捕获.
  [&]       //用到的任何外部变量都隐式按引用捕获
  [=]       //用到的任何外部变量都隐式按值捕获
  [&, x]    //x显式地按值捕获. 其它变量按引用捕获
  [=, &z]   //z按引用捕获. 其它变量按值捕获
  [this]    //通过引用捕获对象，其实就是复制指针
	[*this]   //复制对象来捕获
*/
```

​		定义了一个lambda表达式之后，在编译的时候，编译器会生成一个匿名的类，在这个类中会默认实现一个public类型的operator()函数。所以lambda函数可以直接用 ```lambda(params)``` 来进行调用。所以本质上，lambda函数是一个class/struct。

​		同样，因为在自动生成的类中，可以使用变量的引用来从外部获取变量值，在[capture] 中的 &param 就是会在初始化的时候以引用的方式传值。

​		复制捕获和引用捕获最大的区别就是能不能在匿名函数内改变变量的值。在默认情况下，运算符() 的重载拥有const属性，所以不能改变复制过来的变量。如果想改变复制捕获的变量，要使用mutable关键字。



* 其实最常见的用法就是lambda表达式作为一个函数对象传入别的函数中。



### Closure（闭包）

​		这个名字简直有毒，让人抓不到重点。一般来说closure是指一个带有状态变量的函数，其实也就类似于类的概念了。类的成员函数就是计算部分，类的成员变量就是用来存储状态的。所以能把一段函数代码和一组变量捆绑就形成一个闭包。

​		函数+状态=>闭包，所以闭包可以认为是一个函数对象。

​		

### 模板函数（Template）

首先，模板函数中 ```class``` 完全等同于 ```typename```，两个一样租用的关键字就是历史原因。```template <class T>``` 等同于 ```template <typename T>```

```typename``` 关键字的作用是告诉编译器后面的字符串是一个变量名，消除歧异。使用template/typedef 的时候变量的数据类型要在运行的时候才能确定，在多层依赖下会产生歧义。

```
template <typename T>
void fun1() {
	T::iterator * t; // * 会引起歧义
}

void fun1() {
	typename T::iterator * t; // 表明T::iterator只是一个变量的名字 -> 是一个指针变量
}

struct s1 {
	struct iterator {
		//...
	}
}

struct s2 {
	static int iterator;
}

```

上述代码在使用是可以想到```*``` 可以被认为是指针或者是乘号，如果添加typename 关键字，则表明后面的是类型名字，不是一个成员变量




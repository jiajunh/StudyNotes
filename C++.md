# C++



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

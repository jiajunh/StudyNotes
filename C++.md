# C++

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

   
##Cmake

* 使用```DCMAKE_BUILD_TYPE```设定为debug / release, 显示/不显示 debug 信息

> ```cmake -DCMAKE_BUILD_TYPE = Debug```

* 有时需要指定cmake最小版本，cmake 3.16 及之后支持 precompiled head

> ```cmake_minimum_required(VERSION 3.16)```

* 使用cmake message来输出状态/变量信息

> ```message( [STATUS|WARNING|AUTHOR_WARNING|FATAL_ERROR|SEND_ERROR] "message to display")```

* 对于多平台cmake , 经常需要判断OS

> ```if ([APPLE / UNIX / WIN32])```

* 使用 ```project(demo)```来创建名为```demo```的项目，它会引入两个变量 ```demo_BINARY_DIR``` 和 ```demo_SOURCE_DIR```。同时，cmake 自动定义了两个等价的变量, ```PROJECT_BINARY_DIR``` 和 ```PROJECT_SOURCE_DIR```分别为文件夹目录和根目录

* ```option(<variable> "<help_text>" [value])```
  value: ON/OFF, default=OFF

* ```set(ENV{<variable>} [<value>])```
  设定环境变量
  eg. 设定C++使用版本, 
  	```set (CMAKE_CXX_STANDARD 17) ``` # 指定C++版本为C++17
  	```set (CMAKE_CXX_STANDARD_REQUIRED ON) ``` # 设定为 ON 时 需要设置 CXX_STANDARD
  	```set (CMAKE_CXX_EXTENSIONS OFF)```  # 默认为ON ，若为ON，有些编译器编译时会添加 ```-std=gnu++11``` instead of ```-std=c++11 ```

* 把多个文件存进一个变量, 可用于添加多个.cpp文件到 executable

  ```
  file(GLOB <variable>[LIST_DIRECTORIES true|false]
  		[RELATIVE <path>] [CONFIGURE_DEPENDS]
  		[<globbing-expressions>...])
  file(GLOB_RECURSE <variable> [FOLLOW_SYMLINKS]
  		[LIST_DIRECTORIES true|false] [RELATIVE <path>]
  		[CONFIGURE_DEPENDS]
  		[<globbing-expressions>...])
  ```

  或者

  ```
  aux_source_directory(. SRC_LIST) # 搜索当前目录下的所有.cpp文件
  add_library(demo ${SRC_LIST})
  ```

* ```
  add_executable(<name> source1 source2 ....) include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ```

* 添加外部库文件, private：只在target中用到items中的头文件，public：可能在其他文件中用到items中的头文件，system interface：

  ```
  target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
  ```

* 添加外部库为静态库, 选择使用STATIC 建立静态库， SHARED 建立动态库

  ```
  add_library(<name> [STATIC | SHARED | MODULE]
              [EXCLUDE_FROM_ALL]
              [<source>...])
  ```

* 使用cmake将变量填写到外部系统文件

  ```
  configure_file(<input> <output>
                 [NO_SOURCE_PERMISSIONS | USE_SOURCE_PERMISSIONS |
                  FILE_PERMISSIONS <permissions>...]
                 [COPYONLY] [ESCAPE_QUOTES] [@ONLY]
                 [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
                 
  ```

  之后可以在头文件中, cmake中的变量会填充到```config_name```

  ```#define <var> @config_name@```

  


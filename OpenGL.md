### OpenGL

* 自己编译GLAD和GLFW的时候发现应该先include glad，否则需要更改glfw的makefile，保证没有引入两个一样的库报错

  ```
  #include <glad/glad.h>
  #include <glew/glew3.h>
  ```

  
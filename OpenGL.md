### OpenGL

* 自己编译GLAD和GLFW的时候发现应该先include glad，否则需要更改glfw的makefile，保证没有引入两个一样的库报错

  ```
  #include <glad/glad.h>
  #include <glew/glew3.h>
  ```

* 主要有vertexShader，fragmentShader，定义完之后绑定到一个shaderProgram，就完成了一个着色器

  > shader 需要用GLSL来写，通过读取string 来编译使用
  >
  > Shader是GPU程序，可以使用 uniform 变量从cpu发信息到gpu程序中

* 


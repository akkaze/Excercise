from distutils.core import setup, Extension
echomodule = Extension("echo",
                        sources = ["echo.c"])
setup(name = "echo",
        version = "1.0",
        description = "test",
        author = "dudu",
        ext_modules = [echomodule])

#include <Python.h>
PyObject* echo(PyObject* self, PyObject* args)
{
        char* input = NULL;
        if(!PyArg_ParseTuple(args, "s", &input))
        {
                printf("parse arg errorn");
                return NULL;
        }

        int count = 0;
        do
        {
                printf("%sn", input);
                count++;
        }while(count < 100);
        return Py_BuildValue("i", 0);
}

static PyMethodDef EchoMethods[] =
{
        {"echo", (PyCFunction)echo, METH_VARARGS},
        {NULL, NULL}
};

PyMODINIT_FUNC initecho()
{
        Py_InitModule("echo", EchoMethods);
}

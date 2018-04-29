all: cmodule.so dranxor2/dranxor2C.o

cmodule.so: c_module.c c_module.h cmodule.pyx  dranxor2/dranxor2C.o
	python setup.py build_ext --inplace 

dranxor2/dranxor2C.o : dranxor2/dranxor2C.f dranxor2/dranxor2C.h
	gfortran dranxor2/dranxor2C.f -O3 -fPIC -c -o $@

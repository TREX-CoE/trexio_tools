.SUFFIXES: .f90 .F90

FC=ifort 
LIBS=-ltrexio

.f90.o:
	$(FC) -c $< -o $@

.F90.o:
	$(FC) -c $< -o $@

.PHONY: all

all: trexio_to_fcidump


trexio_to_fcidump: trexio_to_fcidump.o trexio_module.o
	$(FC) $^ -o $@ $(LIBS)

trexio_to_fcidump.o: trexio_module.o


.SUFFIXES: .f90

FC=gfortran -g
LIBS=-ltrexio

.f90.o:
	$(FC) -c $< -o $@

.PHONY: all

all: trexio_to_fcidump


trexio_to_fcidump: trexio_to_fcidump.o trexio_f.o
	$(FC) $^ -o $@ $(LIBS)

trexio_f.o: trexio_f.f90
trexio_to_fcidump.o: trexio_to_fcidump.f90 


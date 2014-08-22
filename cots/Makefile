#################################                                                                                                                                                 
#  Compiler, tools and options  #
#################################

CXX = g++ 
DEBUG = -g
CFLAGES = -Wall -c $(DEBUG)
LFLAGES = -Wall $(DEBUG)
LINK = g++ 
CXXFLAGES = -O3 -DADD_
MC = mpicc
MCXX=mpicxx
BLASPATH = /opt/atlas/
INCLUDE = $(BLASPATH)include
CBLIB = $(BLASPATH)lib/libcblas.a $(BLASPATH)lib/libatlas.a
LIB1=./lib/libcblas.a
LIB2=./lib/libatlas.a
####### Compile
all: main runtest

main: 
#test:
	$(MCXX) $(CFLAGES) main.cpp -I $(INCLUDE)
	$(MCXX) $(CFLAGES) load.cpp
	$(MCXX) -o main main.o load.o $(LIB1) $(LIB2) -lm

runtest:
	scp main root@gpu-server2:/home/common/crd/cots
	mpirun_rsh -np 44 -hostfile hostfile ./main
    
cleanall:
	rm -f *.o main

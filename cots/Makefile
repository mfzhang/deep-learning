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


showthree: showthree.cpp
	$(CXX) $(CFLAGES) showthree.cpp
	$(LINK) -o showthree showthree.o `pkg-config --libs opencv`

showweight: showweight.cpp
	$(CXX) $(CFLAGES) showweight.cpp
	$(LINK) -o showweight showweight.o `pkg-config --libs opencv`

show: show.cpp
	$(CXX) $(CFLAGES) show.cpp
	$(LINK) -o show show.o `pkg-config --libs opencv`

svm: svmclassify.cpp
	$(CXX) $(CFLAGES) svmclassify.cpp
	$(LINK) -o svm svmclassify.o `pkg-config --libs opencv`

cleanall:
	rm -f *.o show svmclassify showthree showweight

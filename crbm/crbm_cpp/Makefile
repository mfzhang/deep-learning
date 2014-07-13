#################################                                                                                                                                           
#  Compiler, tools and options  #
#################################

CXX = g++ 
DEBUG = -g
CFLAGES = -Wall -c $(DEBUG)
LFLAGES = -Wall $(DEBUG)
LINK = g++ 
CXXFLAGES = -O3 -DADD_

####### Compile

test: 
	$(CXX) $(CFLAGES) test.cpp
	$(CXX) $(CFLAGES) matrix.cpp 
	$(LINK) -o test test.o matrix.o
	

clearall:
	rm -f *.o test

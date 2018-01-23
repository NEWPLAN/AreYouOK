# ==============================================================================
# Copyright 2017 NEWPLAN. All Rights Reserved.
#
# Just for fun :).
#
# ==============================================================================
CC = g++  
WARNING=  -Wall
CFLAGS += $(WARNING) -g  -std=c++11 -Os
LDFLAGS += -lm  -pthread
PROC_NAME += areyouok
HEAD_DEPEND += 
SOURCE +=  areyouok.cpp
OBJS_DEPEND=$(subst .cpp,.o, $(SOURCE))
all: clean $(PROC_NAME)
	@echo 'make done'

# genetate objs
%.o:%.cpp $(HEAD_DEPEND)
	$(CC) -c $(CFLAGS) $< -o $@

$(PROC_NAME):$(OBJS_DEPEND)
	$(CC) $^ -o $@ $(LDFLAGS)

run:all
	./$(PROC_NAME)

.PHONY:clean
clean:  
	rm -rf $(OBJS_DEPEND) 
	rm -rf ${PROC_NAME}


CC = g++  
WARNING=  -Wall
CFLAGS += $(WARNING) -g  -std=c++11 -Os
LDFLAGS += -lm  -pthread
PROC_NAME += proc
HEAD_DEPEND += bcube_comm.h bcube_message.h bcube_ops.h bcube_utils.h
SOURCE +=  bcube_comm.cpp bcube_message.cpp bcube_ops.cpp bcube_test.cpp bcube_utils.cpp
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


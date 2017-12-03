CC = g++  
WARNING=  -Wall
CFLAGS += $(WARNING) -g  -std=c++11 -Os
LDFLAGS += -lm  -pthread
PROC_NAME += proc
#OBJS_DEPEND += Switch.o Server.o Link.o Parameter.o BCube.o Utils.o Main.o Message.o
HEAD_DEPEND += Utils.h Message.h Bcube_ops.h Bcube.h
SOURCE =  Utils.cpp Main.cpp Message.cpp Bcube_ops.cpp Bcube.cpp
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
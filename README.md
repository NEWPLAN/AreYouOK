## RDMA test

RDMA Test is an simple example for rdma communication technology.

Here we user a client/server topology to test whether there are something wrong.


How to play:

firstly:
	Make sure you have installed a rdma dirver like OFED or other dirvers.
	Make sure your hardware is supported for RDMA like Mellanox Infinibands.

then compile it with: 
	cd 'path to Makefile' && make

run server part:
	./server


run client part:
	./client 'ip to server' <filename>

	here 'ip to server' is necessary point to server nodes
	and you can fill the filename if you want to send a file to server nodes
	
notes:
	here we have test for 40Gbps switch and nic with 8 client ---> 1 server test.
	

NEWPLAN

2017.12.6

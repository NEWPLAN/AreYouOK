## Bcube Topology for Tensorflow Gradient update

This work is main to redesign a new decentralized parameter gradient update. 
with the advantage of Bcube topology.


install:

firstly, if you have a cuda environment, you need to enable HAVE_CUDA

then run command like 
python setup.py build && sudo python setup.py install

to make an install

if you want to run a tensorflow code. you need first export system environment like
export BCUBE_RANK=NUM 
in each physical node.
here num is a the rank number, we assume you have a 2 level topology where bcube0 has 3 machines
so all the rank is 9, NUM is in [0,1,2,3,4,5,6,7,8]
and each node have two nic, you need specific you ip address before run

if all the request is finished, you can run the example with 
python examples/tensorflow_mnist.py



NEWPLAN

2017.12.6

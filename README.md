# A Very Simple Example of NCCL Backend Distributed Pytorch

Running a simple pytorch all reduce operation across X nodes example.

Ensure that the correct IP address is given for the --dist-url argument inside the script.

On Node 0 : 
```
python all_reduce.py --rank 0 --world-size 2 --dist-url tcp://masternode-ip:freeport
```

On Node 1:

```
python all_reduce.py --rank 1 --world-size 2 --dist-url tcp://masternode-ip:freeport
```


Change the world-size, rank appropriately based on the node.
For example, if you want to run this job on 10 nodes, the world-size would be 10. Ranks would be from 0<=x<=9.

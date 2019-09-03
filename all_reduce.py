# This script demos the concept of sitributed training with pytorch using nccl as the backend.
# Does a distributed all reduce across 2 nodes using all the GPUs in the node.
# The output across all the tensors after all reduce operation must be equal to the number of GPUs across all the nodes.

import torch
import torch.distributed as dist 
import argparse


parser = argparse.ArgumentParser(description='PyTorch Simple Distributed Training')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://gpu002:9001', type=str,
                    help='url used to set up distributed training.' 
                         'This is the url to the master node. This node co-ordinates the overall training process.')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()
    dist.init_process_group(backend=args.dist_backend,
    			init_method=args.dist_url,
    			world_size=args.world_size,
    			rank=args.rank)
    
    tensor_list = []
    for dev_idx in range(torch.cuda.device_count()):
        tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
    
    print("Before All Reduce: ")	   
    print("Using: " + str(len(tensor_list)) + " tensors as follows:")
    print(tensor_list)
    print("Performing All Reducing Sum OP: ")	   
    result = dist.all_reduce_multigpu(tensor_list, async_op=True)
    print(result)
    #print(type(result))i
    print("\n Results after All Reduce OP: ")	   
    print(tensor_list)

if __name__=='__main__':
    main()



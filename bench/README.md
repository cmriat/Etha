# Test results
Test on 16 GPUs with 2 nodes.
source mesh placement: [Replicate(), Shard(0), Replicate(), Shard(dim=1)]
target mesh placement: [Replicate(), Replicate(), _StridedShard(0), Shard(0)]
* 1, 1, 4, 2 -> 4, 2, 1, 1
![Mesh 1](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1.png)
* 4, 1, 1, 2 -> 2, 1, 1, 4
![Mesh 2](./results/throughput_benchmark_mesh_01_4_1_1_2_2_1_1_4.png)
* 4, 1, 2, 1 -> 1, 1, 1, 8
![Mesh 3](./results/throughput_benchmark_mesh_02_4_1_2_1_1_1_1_8.png)
* 4, 2, 1, 1 -> 1, 1, 2, 4
![Mesh 4](./results/throughput_benchmark_mesh_03_4_2_1_1_1_1_2_4.png)
* 2, 1, 4, 1 -> 1, 4, 1, 2
![Mesh 5](./results/throughput_benchmark_mesh_03_2_1_4_1_1_4_1_2.png)
* 2, 2, 2, 1 -> 1, 1, 4, 2
![Mesh 6](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2.png)
* 2, 4, 1, 1 -> 1, 2, 4, 1
![Mesh 7](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1.png)
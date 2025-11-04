# Test results
Test on 16 GPUs with 2 nodes.

source mesh placement: [Replicate(), Replicate(), Shard(0), Shard(1)]
target mesh placement: [Replicate(), _StridedShard(1, split_factor=2), Replicate(), Shard(1)]

## Mesh Configuration 1
* 1, 1, 4, 2 -> 4, 2, 1, 1
![Mesh 1](./results/throughput_benchmark_mesh_01_1_1_4_2_4_2_1_1.png)

## Mesh Configuration 2
* 8, 1, 1, 1 -> 1, 2, 2, 2
![Mesh 2](./results/throughput_benchmark_mesh_02_8_1_1_1_1_2_2_2.png)

## Mesh Configuration 3
* 1, 2, 4, 1 -> 8, 1, 1, 1
![Mesh 3](./results/throughput_benchmark_mesh_03_1_2_4_1_8_1_1_1.png)

## Mesh Configuration 4
* 2, 2, 2, 1 -> 1, 1, 4, 2
![Mesh 4](./results/throughput_benchmark_mesh_04_2_2_2_1_1_1_4_2.png)

## Mesh Configuration 5
* 2, 4, 1, 1 -> 1, 2, 4, 1
![Mesh 5](./results/throughput_benchmark_mesh_05_2_4_1_1_1_2_4_1.png)

## Mesh Configuration 6
* 4, 1, 1, 2 -> 2, 1, 1, 4
![Mesh 6](./results/throughput_benchmark_mesh_06_4_1_1_2_2_1_1_4.png)

## Mesh Configuration 7
* 4, 1, 2, 1 -> 1, 1, 1, 8
![Mesh 7](./results/throughput_benchmark_mesh_07_4_1_2_1_1_1_1_8.png)

## Mesh Configuration 8
* 4, 2, 1, 1 -> 1, 1, 2, 4
![Mesh 8](./results/throughput_benchmark_mesh_08_4_2_1_1_1_1_2_4.png)
"""多节点拓扑:节点 IP 发现 + train/infer 角色切分。对齐 rl 框架的约定。

节点 IP 经 headless Service DNS 发现(socket.gethostbyname + 轮询;kjobctl 的
NODELIST DNS 单次 getent 不稳,要重试)。节点按 index 切:前 train_nodes 个训练,
其余推理。cross-world rank:trainer 0.., inference 从 train_nodes*gpus 起。
"""

import os


class Topo:
    """NODE_IPS(逗号分隔)由 sbatch 用 kubectl 发现并导出;节点按 index 切 train/infer。"""

    def __init__(self, train_nodes, infer_nodes, gpus_per_node=8):
        self.train_nodes = train_nodes
        self.infer_nodes = infer_nodes
        self.gpus = gpus_per_node
        self.node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("SLURM_NODEID", 0)))
        self.ips = os.environ.get("NODE_IPS", "127.0.0.1").split(",")

    @property
    def role(self):
        return "train" if self.node_rank < self.train_nodes else "infer"

    @property
    def role_rank(self):  # 本角色内的节点序(torchrun/vllm 的 --node-rank)
        return self.node_rank if self.role == "train" else self.node_rank - self.train_nodes

    @property
    def train_head(self):
        return self.ips[0]

    @property
    def infer_head(self):
        return self.ips[self.train_nodes]

    @property
    def infer_base_rank(self):  # etha cross-world:inference rank 从这里起
        return self.train_nodes * self.gpus

    @property
    def world(self):  # cross-world 总 rank 数
        return (self.train_nodes + self.infer_nodes) * self.gpus

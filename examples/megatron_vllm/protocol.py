"""通用引擎权重接口:每个引擎(训练/推理)实现一份,driver 与 etha 只面向它。

权威清单 = HF checkpoint 的权重名列表(safetensors index)——模型逻辑权重的
第三方全集,两端遍历同一份:顺序免协商(清单序即 canonical 序);"单边独有"的
权重(trainer 的 value head、vLLM 的 kv/weight scale)不在清单上,天然不传;
清单上的名字两端必须都有,查不到 = 映射/部署 bug,raise 而非 skip。
部署导致的子集差异(如 vLLM 未加载 MTP 头)由 driver 裁剪清单一次性解决;
tied embedding 在清单上只有一份,折叠自动完成。

三个方法对应设计文档的四类差异(① name 隐含在 key 里:每端内部维护
HF 名 → 本引擎 param 的映射,对外只说 HF 语言):

  ② parallel —— get_sharding():该逻辑权重的 (mesh_tensor, placements) 声明。
                可序列化(KB 级),跨进程交给对端算 plan;rank 用 cross-world 编号。
  ③ 仿射     —— local_view():该逻辑权重在本进程的 tensor view,本地消费不跨进程。
                源端 = 发送的 shard(含发送前的轴级变换,view 即变换);
                收端 = 落点(直落 kernel param 的 fuse 段;process 非平凡时为 staging)。
  ④ 非仿射   —— process_after_load():仅收端、仅 process 非平凡的配置下非空。
                收一批名字(process 以 module 为单位,调用方保证整 module 落齐)。

driver 的通用流程(不含任何引擎知识):

  init:  清单 = HF index(driver 按部署裁剪一次,下发两端)
         两端声明经 driver 互换 → 每端本地:
           for name in 清单:
               m2m = get_m2m_map(*src声明[name], *dst声明[name])
               chunks += m2m_to_chunks(m2m, my_rank,
                                       source_tensor=api.local_view(name),   # 源端
                                       target_tensor=api.local_view(name))   # 收端
         route_idx 全局重编,chunks 缓存
  每轮:  chunk_comm(chunks, group) → 收端 api.process_after_load(清单)
"""

from typing import Protocol
from collections.abc import Iterable

import torch
from torch.distributed.tensor import Placement


class EngineWeightProtocol(Protocol):
    def get_sharding(self, hf_name: str) -> tuple[torch.Tensor, tuple[Placement, ...]]: ...

    def local_view(self, hf_name: str) -> torch.Tensor: ...

    def process_after_load(self, hf_names: Iterable[str]) -> None: ...

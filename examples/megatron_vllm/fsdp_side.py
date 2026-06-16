"""torch-native(FSDP2/DTensor)发端:susser-tod / torchtitan / VeOmni 式。

引擎矩阵里「免费」那格的实证,与 trainer_side.py(Megatron)对照:
那边要 mesh 装配 + placement 白名单 + 名字表 + 去交错 view 四样手工;
这边参数本身就是带 placement 的 DTensor——声明直接读,包括 FSDP2+TP/EP
叠出的 _StridedShard(etha 原生支持);布局贴 HF,仿射 = to_local() 完事。
"""


class FsdpWeightProtocol:
    def __init__(self, model, base_rank=0):
        # torch-native 框架命名贴 HF(susser-tod 的 GroupedMoE 天然 grouped 名);
        # 偏离时一张薄改名表,name 是 trainer→HF 的固有成本,但这里几乎为零。
        self.base_rank = base_rank
        self._params = dict(model.named_parameters())

    def get_sharding(self, hf_name):
        p = self._params[hf_name]
        return self.base_rank + p.device_mesh.mesh, p.placements, p.dtype

    def local_view(self, hf_name):
        return self._params[hf_name].to_local()

    def process_after_load(self, hf_names):
        pass  # 发端无 ④ 非仿射

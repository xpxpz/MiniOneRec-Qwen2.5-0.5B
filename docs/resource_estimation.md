# GPU 资源与成本预估

首选单张 L40S 48 GB。24 GB 是待验证的低成本候选，受 4B 编码器 2048 token 批量、SFT 512 token 和 GRPO 16 beam 全词表 logits 峰值影响。GRPO 的 16 条轨迹须留在同一微批次内，减少梯度累积不能降低该微批次的峰值。

以下为计算预算情景。以论文样本数作为量级输入，吞吐均为假设值。上线后用短 benchmark 的实际 token 长度、数据量和吞吐替换。

| 阶段 | 估算输入与吞吐假设 | 单卡 GPU-hours |
|---|---|---:|
| 商品语义编码 | 3,686 个商品，每秒 2–20 个 | 0.05–0.52 |
| RQ-VAE 及 SID | 10,000 个全批训练步，每步 0.05–0.30 秒，另留初始化、评估和生成余量 | 0.2–1.2 |
| SFT | 按 80,000 样本/epoch 的预算情景、10 epoch、25–100 样本/秒 | 2.2–8.9 |
| GRPO | 按 53,000 prompt/epoch、2 epoch、每秒 0.5–3 prompt，含每组 16 候选及反向 | 9.8–58.9 |
| 约束评估 | 4,533 条测试样本，每秒 1–10 条，每条 50 beam | 0.13–1.26 |

合计约 12.4–70.8 GPU-hours。加 20% 时间余量，预算约 15–85 租用小时。按 Runpod L40S Secure Cloud 的 $1.09/h 示例，约 $16–93 GPU 租金，另按账单汇总存储、网络和税费。上述为训练前预算情景，不代表实测。正式阶段已记录的设备时间及费用信息见 [复现报告](reproduction_report.md)；实际云费用尚未取得账单，不能认定为零。 [Runpod 公开参考费率](https://www.runpod.io/articles/guides/ai-server-cost)

服务器环境为 Linux x86_64、Python 3.11、uv、匹配 CUDA 12.4 PyTorch wheel 的 NVIDIA 驱动。环境同步用 `uv sync --locked`。建议预留至少 60 GB 磁盘以容纳模型、数据、优化器状态、合并权重及多次阶段检查点。完成正式 GPU 阶段后使用节点账单和 `outputs/logs/stages.jsonl` 对账。显存字段为每秒采样的设备总占用峰值。


正式运行环境与上面的预算环境不同：服务器使用 Python 3.12.3 和厂商 PPU PyTorch 2.6.0 / CUDA 12.6，而仓库 uv 锁定环境要求 Python 3.11 并针对 NVIDIA CUDA 12.4。两者不是同一环境，重新复现时需先按目标设备解决驱动和依赖兼容性；参见正式报告。

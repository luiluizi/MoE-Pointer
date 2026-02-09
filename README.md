# Multi-Agent Pointer Transformer


# 实验记录

```
scenario: small
interruptable: False
use_sp: True

n_frame: 58
n_requested_frame: 42
n_vehicle: 5
n_node: 20

n_init_requests: 10
n_norm_requests: 100
max_consider_requests: -1

max_capacity: 3
dist_distribution: random
max_dist: 10
dist_cost: 0.0

info_batch_size: 0
debug: False
```

`OR-Tools 只测了4个样例，规划周期为10帧，初始帧由于请求少是瓶颈`
|small|obj.|time|service rate|
|-|-|-|-|
|OR-Tools|238.0|13 min|0.79|
|Heuristic|158.0|< 1 s|0.48|
|Ours|249.4|2 s|0.76|

|small_2D|obj.|time|service rate|
|-|-|-|-|
|OR-Tools|256.5|13 min|0.79|
|Heuristic|163.0|< 1 s|0.50|
|Ours|241.0|2 s|0.75|

|small_cost|obj.|time|service rate|
|-|-|-|-|
|OR-Tools|135.0|16 min|0.77|
|Heuristic|66.8|< 1 s|0.48|
|Ours|102.2|2 s|0.73|

# TODO

- [x] 考虑 cost。如果只考虑吞吐 reward，那么需要问题是运力 bound，而请求累积导致转换为静态问题。并且要放宽请求 no_action 的概率
- [x] Scale Up
- [ ] Credit 分配
- [ ] 研究一下优势分解引理
- [ ] 在启发式概率中加入 bias，不至于出现 0 或 1 这样的极端
- [x] 更强的启发式引导：车辆需要根据节点上物品与自己携带物品的终点相融性来确定启发值，也要根据距离判断
- [ ] 节点太多导致车辆调度即任务分配
- [x] assign 时要保证 tsp 结果不会超出时间限
- [ ] 加入 Dropout
- [ ] beam search 采样
- [ ] Immitation Learning
- [x] enlarge entropy coefficiency
- [ ] 将 heuristic probability 在一开始就输入神经网络
- [ ] 加入 hindsight 策略
- [x] 将 qkfeat 融入之前再加一个线性层，否则梯度更新过于混乱
- [x] Add dist distribution support.
- [ ] Relation-Aware Self Attention 为什么不起作用
- [ ] 调整原地 stay 的概率超参
- [ ] 如果是可以提前分配请求的话，可以避免两辆车前往同一个地点。也就是车辆的解码结果是请求，而不是节点。但是这样的话，无法做reposition。在启发值时，如果已经有车辆去往这个点了，那么后续车辆解码时应该抑制该点。
- [x] 复现 MAPDP。
- [ ] 先任务分配再 Dispatch
- [x] 为什么神经网络无法感知到最后时刻要优先配送
- [ ] 任务分配 + 单独规划
- [x] support node embedding.
- [ ] 大数据时需要调小学习率，多 warmup，调小 clip_ratio
- [ ] dhrd 数据集存在过拟合问题
- [x] fix TSP heuristic bug
- [x] 随机性 bug
- [x] 感觉 entropy 会不断增大
- [x] 启发式概率不计入梯度运算。试过了，效果不好。
- [ ] 支持时间窗口
- [ ] 动量太大导致训练不稳定？
- [ ] get_obs 时优先返回有价值的请求
- [ ] 针对涨不动，应该调
- [ ] 从 sample 转为 argmax 是越训越差的原因吗？
- [x] 测试一下不 reset seed 的效果。
- [ ] Softmax 尺度，梯度消失问题。
- [ ] softmask_loadbalance_probs + softmask_delivery_diversity_probs，进行标准化，不同数据集尺度不同。
- [ ] 使用 Heuristic Prior 会对最终的 Softmax 产生梯度消失的影响吗？仔细想一下这个尺度？

``` latex
% 画一个 curve
% 手动生成数据分布
% 验证不同城市之间有 pattern
% 看一下 Attention Map，深度研究一下
% 看一下领域内有哪些 baseline
```

# Discusions

- 由于是以 Node 为目标的调度，不是以 Request 为目标的调度，所以在 Softmax 之后，目标 Node 会受到大量无关 Node 的影响，导致模型根本训不动。

#

```bash
python3 train_mvdpdp.py --algorithm_name=or --only_eval --eval_episodes=1
python3 train_mvdpdp.py --env_config_path=envs/mvdpdp/env_args_dhrd.yaml --mini_batch_size=32 --n_rollout_threads=32 --lr=1e-5
```
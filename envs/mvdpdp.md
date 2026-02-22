# Multi Vehicle Dynamic Pickup Delivery Problem, MVDPDP, 动态多车取送问题

## Solvers

| Abbreviation                                                                            | Full Name                                        |
| --------------------------------------------------------------------------------------- | ------------------------------------------------ |
| [copt](https://www.shanshu.ai/)                                                            | MVDPDP 建模为整数规划较复杂                      |
| [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver?hl=zh-cn) | 感觉对离散问题更友好，不需要完全建模为整数规划。 |

## Abbreviations

| Abbreviation | Full Name                                     |
| ------------ | --------------------------------------------- |
| MVDPDP       | Multi Vehicle Dynamic Pickup Delivery Problem |
| RL           | Reinforcement Learning                        |

## background

本工作是用 RL 来做 MVDPDP，因为决策时未来的 requests 未知，所以 RL 理论上达不到最优解。

现在的仿真环境的实现方式是，所有的 Requests 在初始化时就生成完成，但是随着时间才逐渐可见；也就是说未来 Requests 与当前的状态无关（因为 Requests 是早就生成好的），这在生活中也比较常见，比如外卖订单的分布和当前棋手的分布是无关的。

现在为了评估 RL 策略的好坏，我们希望知道理论最优解。我们通过预知未来的方式提前获取所有的 requests 以及其出现时刻，由此将动态问题转化为静态问题，然后使用传统求解器来求解。

MVDPDP 可以分为若干版本，其中 `离散空间，离散时间，不可中断` 版本是最容易仿真的一个，我们优先在这个版本上验证算法的有效性。

### 空间

- 离散空间：$N$ 个节点，两两之间有一个距离。每个请求为从一个点去往另一个点。车辆在节点间行驶的过程中不存在具体的位置。
- 连续空间：$N$ 个节点，二维平面上采样
- 路网空间：真实路网。车辆在节点间行使的过程中有在路网上具体的位置。

### 时间

- 离散时间：节点间行驶的时间，请求出现的时间都是整数。
- 连续时间：节点间行驶的时间，请求出现的时间为不是整数。

### 中断

- 不可中断：从一个点去往另一个点的过程中不可中断，比如滴滴司机被预定后，只能去接该乘客，就算在这过程中出现其他更优的乘客也不行。
- 可中断：从一个点去往另一个点的过程中可中断，比如仓储机器人取物，随着请求的到来，可以重新规划每个机器人的目标以实现更优解。

## 时间离散，空间离散，不可中断

### Instance Variable

| Symbolic                                               | Explaination               |
| ------------------------------------------------------ | -------------------------- |
| N                                                      | 节点数                     |
| M                                                      | 请求数                     |
| K                                                      | 车辆数                     |
| T                                                      | 帧数，不算初始帧           |
| $start_i \in [0, N), i \in [0, K)$                   | 每辆车的起点               |
| $capacity_i, i \in [0, K)$                           | 每辆车的容量               |
| $dist_{i,j}, i \in [0, N), j \in [0, N)$             | 节点距离矩阵，节点时间矩阵 |
| $from_i \in [0, N), \;to_i \in [0, N), \;i\in[0, M)$ | 请求的起点和终点           |
| $apper_i \in [0, T], i \in [0, M)$                   | 请求的出现时间             |

### Input Constraints

| Symbolic                          | Explaination   |
| --------------------------------- | -------------- |
| $from_i \ne to_i, \;i\in[0, M)$ | 起点和终点不同 |

### Decision Variable

`我的建模不一定正确，你得去网上搜一下别人怎么建模。另一种建模方式是t表示操作序列，不表示时间`

| Symbolic                                                | Explaination                                         |
| ------------------------------------------------------- | ---------------------------------------------------- |
| $pickup_{i, j, t}\in\{0,1\}, i\in[0,K), j\in[0, M)$   | $vehicle_i$ 在时刻 $t$ pickup 了 $request_j$   |
| $delivery_{i, j, t}\in\{0,1\}, i\in[0,K), j\in[0, M)$ | $vehicle_i$ 在时刻 $t$ delivery 了 $request_j$ |

### Constraints

`我不一定考虑全了，可能有其它的约束`

| Constraint                                                                                         | Explaination                                                                                    |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
|                                                                                                    | 每个 request 最多只能被一辆车 pickup，最多被一辆车 delivery                                     |
|                                                                                                    | 如果某个 request 被 delivery 了，那么应该被**同**一辆车 pickup                            |
| $any_{i, t}(delivery_{i,j,t}) \rightarrow argmax_t(pickup_{i,j,t}) < argmax_t(delivery_{i,j,t})$ | 如果某个 request 被 delivery 了，那么它也应该被 pickup 了，并且 pickup 时间要小于 delivery 时间 |
|                                                                                                    | 每个时刻每辆车的载荷不能超过其容量                                                              |
|                                                                                                    | 每个时刻，每辆车，pickup 和 delivery 的地点应该相同                                             |
|                                                                                                    | 每辆车，相邻两次操作（pickup/delivery）之间的时间应该小于两地$dist_{i,j}$                     |
|                                                                                                    | pickup 时间应该小于等于 request 出现时间                                                        |

### Objective

比如最大化**有效**行驶距离总和

最小化未被 delivery 的 request 数量

最小化行驶距离总和 + 未被 delivery 的 request 的 penalty

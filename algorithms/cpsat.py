from ortools.sat.python import cp_model
import numpy as np

from .simulated_annealing import SimulatedAnnealing

class CPSAT(cp_model.CpModel):
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=4.0, courier_stage1_temp=None, objective_scale=1., env_args=None, **kwargs):
        """
        :param N: 节点数
        :param M: 请求数
        :param K: courier数
        :param D: drone数
        :param T: 时间帧数
        :param start_K: 每个courier的起点
        :param start_D: 每个drone的起点
        :param capacity: 每辆车的容量
        :param join_time_K: 每个courier开始工作的时间，use for courier whose time_left != 0
        :param join_time_D: 每个drone开始工作的时间，use for drone whose time_left != 0
        :param dist: 距离矩阵
        :param cost: 花费矩阵
        :param from_req: 每个请求的起点
        :param to_req: 每个请求的终点
        :param station1_req: 每个请求的station1
        :param station2_req: 每个请求的station2
        :param appear: 每个请求的出现时间
        :param value: 每个请求的价值
        :param penalty: 每个请求的惩罚
        :param pre_load_K_stage1: K x M，表示请求是否已经在stage1装载在了courier上
        :param pre_load_K_stage3: K x M，表示请求是否已经在stage3装载在了courier上
        :param pre_load_D: D x M，表示请求是否已经装载在了drone上
        :param wait_stage2: M，表示请求是否在等待进入stage2
        :param wait_stage3: M，表示请求是否在等待进入stage3
        :courier_stage1_temp: 记录完成每个订单stage1配送的courier，为了防止stage1和3使用同一个courier
        """
        super().__init__()

        # 常量定义
        if env_args is not None:
            self.max_dist = env_args["max_dist"] * 2
            self.max_cost = round(env_args["max_dist"] * 2 * env_args["dist_cost_drone"] / objective_scale)
        else:
            self.max_dist = 1000
            self.max_cost = 1000
        
        self.N = N
        self.M = M
        self.K = K
        self.D = D
        self.T = T
        self.start_K = start_K
        self.start_D = start_D
        self.capacity = capacity
        self.join_time_K = join_time_K
        self.join_time_D = join_time_D
        self.dist = dist
        
        self.from_req = from_req
        self.to_req = to_req
        self.station1_req = station1_req
        self.station2_req = station2_req
        self.appear = appear
        self.value = value
        self.penalty = penalty
        
        self.pre_load_K_stage1 = pre_load_K_stage1
        self.pre_load_K_stage3 = pre_load_K_stage3
        self.pre_load_D = pre_load_D
        self.wait_stage2 = wait_stage2
        self.wait_stage3 = wait_stage3
        self.drone_speed_ratio = drone_speed_ratio
        self._courier_stage1_temp = courier_stage1_temp
        
        self.objective_scale = objective_scale
        self._dist = dist
        self._cost_K = cost_K
        self._cost_D = cost_D
        
        # 初始化决策变量
        self.pickup1 = {}
        self.pickup2 = {}
        self.pickup_d = {}
        
        self.delivery1 = {}
        self.delivery2 = {}
        self.delivery_d = {}
        # 初始化中间变量
        self.load_k = {}
        self.load_d = {}
        self.location_k = {}
        self.location_d = {}
        self.no_action_k = {}
        self.no_action_d = {}
        self.req_cur_stage = {}
        self.courier_stage1_temp = {}
        self.sa_solver = SimulatedAnnealing(N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio, courier_stage1_temp=courier_stage1_temp)  
        
    def _initialize_variables(self):
        self.dist = [self.NewIntVar(0, self.max_dist, f'dist_{i}_{j}') for i in range(self.N) for j in range(self.N)]
        # 可优化
        self.dist_d = [self.NewIntVar(0, np.ceil(self.max_dist / self.drone_speed_ratio).astype(int), f'dist_{i}_{j}') for i in range(self.N) for j in range(self.N)]
        self.cost_K = [self.NewIntVar(0, self.max_cost, f'cost_k_{i}_{j}') for i in range(self.N) for j in range(self.N)]
        self.cost_D = [self.NewIntVar(0, self.max_cost, f'cost_d_{i}_{j}') for i in range(self.N) for j in range(self.N)] 
        
        dist = self._dist
        cost_K = self._cost_K
        cost_D = self._cost_D
        
        # dist约束
        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.dist, dist[i][j])
        
        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.dist_d, np.ceil(dist[i][j] / self.drone_speed_ratio).astype(int))

        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.cost_K, cost_K[i][j])
        
        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.cost_D, cost_D[i][j])
        
        _pickup1, _pickup2, _pickup_d, _location_k, _location_d = self.sa_solver.solve(show=False, max_iter=10)
        for k in range(self.K):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.pickup1[(k, m, t)] = self.NewBoolVar(f'pickup1_{k}_{m}_{t}') # pickup[(*, *, T)] 没用
                    self.delivery1[(k, m, t)] = self.NewBoolVar(f'delivery1_{k}_{m}_{t}') # delivery[(*, *, -1/0)] 没用

                    self.pickup2[(k, m, t)] = self.NewBoolVar(f'pickup2_{k}_{m}_{t}') # pickup[(*, *, T)] 没用
                    self.delivery2[(k, m, t)] = self.NewBoolVar(f'delivery2_{k}_{m}_{t}') # delivery[(*, *, -1/0)] 没用
                    # Hint: 只对 True 的情况设置 Hint，减少对搜索空间的限制
                    if t >= 0:
                        if m in self.sa_solver.best_solution_K[k][t]["pickup1"] and not self.sa_solver.pre_load_K_stage1[k][m]:
                            self.AddHint(self.pickup1[(k, m, t)], True)
                        # 不设置 False 的 Hint，让 CP-SAT 自由探索
                        if m in self.sa_solver.best_solution_K[k][t]["delivery1"]:
                            self.AddHint(self.delivery1[(k, m, t)], True)
                        
                        if m in self.sa_solver.best_solution_K[k][t]["pickup2"] and not self.sa_solver.pre_load_K_stage3[k][m]:
                            self.AddHint(self.pickup2[(k, m, t)], True)
                        if m in self.sa_solver.best_solution_K[k][t]["delivery2"]:
                            self.AddHint(self.delivery2[(k, m, t)], True)

        for d in range(self.D):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.pickup_d[(d, m, t)] = self.NewBoolVar(f'pickup_d_{d}_{m}_{t}') # pickup[(*, *, T)] 没用
                    self.delivery_d[(d, m, t)] = self.NewBoolVar(f'delivery_d_{d}_{m}_{t}') # delivery[(*, *, -1/0)] 没用
                    # Hint: 只对 True 的情况设置 Hint，减少对搜索空间的限制
                    if t >= 0:
                        if m in self.sa_solver.best_solution_D[d][t]["pickup"] and not self.sa_solver.pre_load_D[d][m]:
                            self.AddHint(self.pickup_d[(d, m, t)], True)
                        if m in self.sa_solver.best_solution_D[d][t]["delivery"]:
                            self.AddHint(self.delivery_d[(d, m, t)], True)
        
        for k in range(self.K):
            for t in range(self.T + 1):
                self.load_k[(k, t)] = self.NewIntVar(0, self.capacity[k], f'load_k_{k}_{t}')
                self.location_k[(k, t)] = self.NewIntVar(-1, self.N - 1, f'location_k_{k}_{t}')
                self.no_action_k[(k, t)] = self.NewBoolVar(f'no_action_k_{k}_{t}') # no_action[(*, T)] 没用
                # Hint
                self.AddHint(self.load_k[(k, t)], self.sa_solver.best_solution_K[k][t]["load"])
                if _location_k[k][t] != -1:
                    self.AddHint(self.location_k[(k, t)], _location_k[k][t])
                    self.AddHint(self.no_action_k[(k, t)], False)
                else:
                    self.AddHint(self.no_action_k[(k, t)], True)

        for d in range(self.D):
            for t in range(self.T + 1):
                self.load_d[(d, t)] = self.NewIntVar(0, 1, f'load_d_{d}_{t}')
                # TODO 是否加入站点约束
                self.location_d[(d, t)] = self.NewIntVar(-1, self.N - 1, f'location_d_{d}_{t}')
                self.no_action_d[(d, t)] = self.NewBoolVar(f'no_action_d_{d}_{t}') # no_action[(*, T)] 没用
                # Hint
                self.AddHint(self.load_d[(d, t)], self.sa_solver.best_solution_D[d][t]["load"])
                if _location_d[d][t] != -1:
                    self.AddHint(self.location_d[(d, t)], _location_d[d][t])
                    self.AddHint(self.no_action_d[(d, t)], False)
                else:
                    self.AddHint(self.no_action_d[(d, t)], True)
        
        # 新增
        for m in range(self.M):
            self.req_cur_stage[(m, 0)] = self.NewBoolVar(f'req_{m}_stage_0')
            self.req_cur_stage[(m, 1)] = self.NewBoolVar(f'req_{m}_stage_1')
            is_pre2, is_pre3 = False, False
            for k in range(self.K):
                if self.pre_load_K_stage3[k][m]:
                    self.Add(self.req_cur_stage[(m, 0)] == False)
                    self.Add(self.req_cur_stage[(m, 1)] == True)
                    is_pre2 = True
                    break
            for d in range(self.D):
                if self.pre_load_D[d][m]:
                    self.Add(self.req_cur_stage[(m, 0)] == True)
                    self.Add(self.req_cur_stage[(m, 1)] == False)
                    is_pre3 = True
                    break
            if self.wait_stage2[m]:
                self.Add(self.req_cur_stage[(m, 0)] == True)
                self.Add(self.req_cur_stage[(m, 1)] == False)
            elif self.wait_stage3[m]:
                self.Add(self.req_cur_stage[(m, 0)] == False)
                self.Add(self.req_cur_stage[(m, 1)] == True)
            elif not is_pre2 and not is_pre3:
                self.Add(self.req_cur_stage[(m, 0)] == False)
                self.Add(self.req_cur_stage[(m, 1)] == False)
        
        for k in range(self.K):
            for m in range(self.M):
                self.courier_stage1_temp[(k, m)] = self.NewBoolVar(f'req_{m}_stage1_courier_{k}')
                if self._courier_stage1_temp[m] == k:
                    self.Add(self.courier_stage1_temp[(k, m)] == True)
                else:
                    self.Add(self.courier_stage1_temp[(k, m)] == False)

    def _add_constraints(self):
        # 预设已经在车上的物品
        for k in range(len(self.pre_load_K_stage1)):
            for m in range(len(self.pre_load_K_stage1[0])):
                self.Add(self.pickup1[(k, m, -1)] == self.pre_load_K_stage1[k][m])
        
        for k in range(len(self.pre_load_K_stage3)):
            for m in range(len(self.pre_load_K_stage3[0])):
                self.Add(self.pickup2[(k, m, -1)] == self.pre_load_K_stage3[k][m])
        
        for d in range(len(self.pre_load_D)):
            for m in range(len(self.pre_load_D[0])):
                self.Add(self.pickup_d[(d, m, -1)] == self.pre_load_D[d][m])

        # 中间变量约束1： load限制
        for k in range(self.K):
            for t in range(self.T + 1):
                if t == 0:
                    self.Add(self.load_k[(k, t)] == sum(self.pickup1[(k, m, _t)] for m in range(self.M) for _t in range(-1, 1)) + sum(self.pickup2[(k, m, _t)] for m in range(self.M) for _t in range(-1, 1))) # t <= 0 时一定没有 delivery
                else:
                    self.Add(self.load_k[(k, t)] == self.load_k[(k, t - 1)] + sum(self.pickup1[(k, m, t)] for m in range(self.M)) - sum(self.delivery1[(k, m, t)] for m in range(self.M)) + sum(self.pickup2[(k, m, t)] for m in range(self.M)) - sum(self.delivery2[(k, m, t)] for m in range(self.M)))

        for d in range(self.D):
            for t in range(self.T + 1):
                if t == 0:
                    self.Add(self.load_d[(d, t)] == sum(self.pickup_d[(d, m, _t)] for m in range(self.M) for _t in range(-1, 1))) # t <= 0 时一定没有 delivery
                else:
                    self.Add(self.load_d[(d, t)] == self.load_d[(d, t-1)] + sum(self.pickup_d[(d, m, t)] for m in range(self.M)) - sum(self.delivery_d[(d, m, t)] for m in range(self.M)))
        # 中间变量约束2： location限制
        for k in range(self.K):
            self.Add(self.location_k[(k, 0)] == self.start_K[k])
            self.Add(self.no_action_k[(k, self.join_time_K[k])] == 0) # courier join 时通过 no_action=False 来将其标记为关键点
            for t in range(self.T + 1):
                if t < self.join_time_K[k]:
                    self.Add(self.no_action_k[(k, t)] == 1) # 在courier join 之前需要保证无操作
                if t != 0:
                    self.Add(self.location_k[(k, t)] == self.location_k[(k, t - 1)]).OnlyEnforceIf(self.no_action_k[(k, t - 1)])
                for m in range(self.M):
                    self.Add(self.location_k[(k, t)] == self.from_req[m]).OnlyEnforceIf(self.pickup1[(k, m, t)])
                    self.Add(self.location_k[(k, t)] == self.station1_req[m]).OnlyEnforceIf(self.delivery1[(k, m, t)])
                    self.Add(self.location_k[(k, t)] == self.station2_req[m]).OnlyEnforceIf(self.pickup2[(k, m, t)])
                    self.Add(self.location_k[(k, t)] == self.to_req[m]).OnlyEnforceIf(self.delivery2[(k, m, t)])
        
        for d in range(self.D):
            self.Add(self.location_d[(d, 0)] == self.start_D[d])
            self.Add(self.no_action_d[(d, self.join_time_D[d])] == 0) # drone join 时通过 no_action=False 来将其标记为关键点
            for t in range(self.T + 1):
                if t < self.join_time_D[d]:
                    self.Add(self.no_action_d[(d, t)] == 1) # 在drone join 之前需要保证无操作
                if t != 0:
                    self.Add(self.location_d[(d, t)] == self.location_d[(d, t - 1)]).OnlyEnforceIf(self.no_action_d[(d, t - 1)])
                for m in range(self.M):
                    self.Add(self.location_d[(d, t)] == self.station1_req[m]).OnlyEnforceIf(self.pickup_d[(d, m, t)])
                    self.Add(self.location_d[(d, t)] == self.station2_req[m]).OnlyEnforceIf(self.delivery_d[(d, m, t)])
        
        # 中间变量约束3：no_action设置
        for k in range(self.K):
            for t in range(0, self.T + 1):
                if t == self.join_time_K[k]:
                    continue
                self.Add(self.no_action_k[(k, t)] == 1).OnlyEnforceIf(
                    [self.pickup1[(k, m, t)].Not() for m in range(self.M)] +
                    [self.delivery1[(k, m, t)].Not() for m in range(self.M)] + 
                    [self.pickup2[(k, m, t)].Not() for m in range(self.M)] +
                    [self.delivery2[(k, m, t)].Not() for m in range(self.M)]
                )
                for m in range(self.M):
                    self.Add(self.no_action_k[(k, t)] == 0).OnlyEnforceIf(self.pickup1[(k, m, t)])
                    self.Add(self.no_action_k[(k, t)] == 0).OnlyEnforceIf(self.delivery1[(k, m, t)])
                    self.Add(self.no_action_k[(k, t)] == 0).OnlyEnforceIf(self.pickup2[(k, m, t)])
                    self.Add(self.no_action_k[(k, t)] == 0).OnlyEnforceIf(self.delivery2[(k, m, t)])
        
        for d in range(self.D):
            for t in range(0, self.T + 1):
                if t == self.join_time_D[d]:
                    continue
                self.Add(self.no_action_d[(d, t)] == 1).OnlyEnforceIf(
                    [self.pickup_d[(d, m, t)].Not() for m in range(self.M)] +
                    [self.delivery_d[(d, m, t)].Not() for m in range(self.M)]
                )
                for m in range(self.M):
                    self.Add(self.no_action_d[(d, t)] == 0).OnlyEnforceIf(self.pickup_d[(d, m, t)])
                    self.Add(self.no_action_d[(d, t)] == 0).OnlyEnforceIf(self.delivery_d[(d, m, t)])
        
        # 约束1：每个请求只能被一辆车接送
        for m in range(self.M):
            vars_pickup1 = []
            vars_delivery1 = []
            vars_pickup2 = []
            vars_delivery2 = []
            vars_pickup_d = []
            vars_delivery_d = []
            for k in range(self.K):
                for t in range(-1, self.T + 1):
                    vars_pickup1.append(self.pickup1[(k, m, t)])
                    vars_delivery1.append(self.delivery1[(k, m, t)])
                    vars_pickup2.append(self.pickup2[(k, m, t)])
                    vars_delivery2.append(self.delivery2[(k, m, t)])
            for d in range(self.D):
                for t in range(-1, self.T + 1):
                    vars_pickup_d.append(self.pickup_d[(d, m, t)])
                    vars_delivery_d.append(self.delivery_d[(d, m, t)])
                    
            self.AddAtMostOne(vars_pickup1)
            self.AddAtMostOne(vars_delivery1)
            self.AddAtMostOne(vars_pickup2)
            self.AddAtMostOne(vars_delivery2)
            self.AddAtMostOne(vars_pickup_d)
            self.AddAtMostOne(vars_delivery_d)
                   
            # self.Add(sum(self.pickup1[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)
            # self.Add(sum(self.delivery1[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)
            # self.Add(sum(self.pickup2[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)
            # self.Add(sum(self.delivery2[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)
            # self.Add(sum(self.pickup_d[(d, m, t)] for d in range(self.D) for t in range(-1, self.T + 1)) <= 1)
            # self.Add(sum(self.delivery_d[(d, m, t)] for d in range(self.D) for t in range(-1, self.T + 1)) <= 1)

        # 约束2：若某个request被delivery， 则必须在之前的某个时间被同一辆车pickup
        # for t in range(-1, self.T + 1):  
        #     for m in range(self.M):
        #         for k in range(self.K):
        #             self.Add(sum(self.pickup1[(k, m, _t)] for _t in range(-1, t)) == 1).OnlyEnforceIf(self.delivery1[(k, m, t)])
        #             self.Add(sum(self.pickup2[(k, m, _t)] for _t in range(-1, t)) == 1).OnlyEnforceIf(self.delivery2[(k, m, t)])
        #         for d in range(self.D):
        #             self.Add(sum(self.pickup_d[(d, m, _t)] for _t in range(-1, t)) == 1).OnlyEnforceIf(self.delivery_d[(d, m, t)])
        
        for t in range(-1, self.T + 1):  
            for m in range(self.M):
                for k in range(self.K):
                    prior_pickup1 = [self.pickup1[(k, m, _t)] for _t in range(-1, t)]
                    prior_pickup2 = [self.pickup2[(k, m, _t)] for _t in range(-1, t)]
                    self.AddBoolOr(prior_pickup1).OnlyEnforceIf(self.delivery1[(k, m, t)])
                    self.AddBoolOr(prior_pickup2).OnlyEnforceIf(self.delivery2[(k, m, t)])
                for d in range(self.D):
                    prior_pickup_d = [self.pickup_d[(d, m, _t)] for _t in range(-1, t)]
                    self.AddBoolOr(prior_pickup_d).OnlyEnforceIf(self.delivery_d[(d, m, t)])             
        
        # 约束3：每个时刻每辆车的load不能超过capacity
        # 注意：变量定义时已经设置了上界，此约束在技术上冗余，但保留以增强可读性
        # 如果追求性能，可以注释掉以下约束（变量定义已保证 load_k <= capacity[k]）
        # for k in range(self.K):
        #     for t in range(self.T + 1):
        #         self.Add(self.load_k[(k, t)] <= self.capacity[k])
        
        # for d in range(self.D):
        #     for t in range(self.T + 1):
        #         self.Add(self.load_d[(d, t)] <= 1)

        # 约束4：每辆车，任意两次操作的时间间隔应该大于两地的距离（第一种写法没考虑初始位置到第一次操作的距离，直接使用location枚举效率更高）
        for k in range(self.K):
            for t in range(self.T + 1):
                for t1 in range(t + 1, self.T + 1):
                    index = self.NewIntVar(0, self.N * self.N - 1, f'index_k_{k}_{t}_{t1}')
                    self.Add(index == self.location_k[(k, t)] * self.N + self.location_k[(k, t1)])
                    dis = self.NewIntVar(0, self.max_dist, f'dis_k_{k}_{t}_{t1}')
                    self.AddElement(index, self.dist, dis)
                    self.Add((t1 - t) >= dis).OnlyEnforceIf(self.no_action_k[(k, t)].Not(), self.no_action_k[(k, t1)].Not())

        for d in range(self.D):
            for t in range(self.T + 1):
                for t1 in range(t + 1, self.T + 1):
                    index = self.NewIntVar(0, self.N * self.N - 1, f'index_d_{d}_{t}_{t1}')
                    self.Add(index == self.location_d[(d, t)] * self.N + self.location_d[(d, t1)])
                    # 无人机速度是骑手的drone_speed_ratio倍，所以使用考虑了速度倍数的dist_d
                    max_dist_d = np.ceil(self.max_dist / self.drone_speed_ratio).astype(int)
                    dis = self.NewIntVar(0, max_dist_d, f'dis_d_{d}_{t}_{t1}')
                    self.AddElement(index, self.dist_d, dis)
                    self.Add((t1 - t) >= dis).OnlyEnforceIf(self.no_action_d[(d, t)].Not(), self.no_action_d[(d, t1)].Not())
        
        # 约束5：pickup的时间应该在request出现之后
        for k in range(self.K):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.Add(t >= self.appear[m]).OnlyEnforceIf(self.pickup1[(k, m, t)])
                    self.Add(t >= self.appear[m]).OnlyEnforceIf(self.pickup2[(k, m, t)])
                    
        for d in range(self.D):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.Add(t >= self.appear[m]).OnlyEnforceIf(self.pickup_d[(d, m, t)])

        # 约束6：服务request的courier不能是同一个
        for m in range(self.M):
            for k in range(self.K):
                self.Add(sum(self.pickup2[(k, m, t)] for t in range(-1, self.T + 1)) == 0).OnlyEnforceIf(self.courier_stage1_temp[(k, m)])
        # 约束7：服务要一个阶段一个阶段完成
        # 处理预分配或已完成部分阶段的请求
        # for m in range(self.M):
        #     for i in range(2):
        #         self.Add(sum(self.pickup1[(k, m, t)] for t in range(-1, self.T + 1) for k in range(self.K)) == 0).OnlyEnforceIf(self.req_cur_stage[(m, i)])
        #         self.Add(sum(self.delivery1[(k, m, t)] for t in range(-1, self.T + 1) for k in range(self.K)) == 0).OnlyEnforceIf(self.req_cur_stage[(m, i)])
        #         if i > 0:
        #             self.Add(sum(self.pickup_d[(d, m, t)] for t in range(-1, self.T + 1) for d in range(self.D)) == 0).OnlyEnforceIf(self.req_cur_stage[(m, i)])
        #             self.Add(sum(self.delivery_d[(d, m, t)] for t in range(-1, self.T + 1) for d in range(self.D)) == 0).OnlyEnforceIf(self.req_cur_stage[(m, i)])
        for m in range(self.M):
            for i in range(2):
                if i == 0:
                    # Stage 0: 请求已完成 Stage 1，禁止再次进行 Stage 1 操作
                    for k in range(self.K):
                        for t in range(-1, self.T + 1):
                            self.Add(self.pickup1[(k, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
                            self.Add(self.delivery1[(k, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
                else:
                    # Stage 1: 请求已完成 Stage 2，禁止所有 Stage 1 和 Stage 2 操作
                    for k in range(self.K):
                        for t in range(-1, self.T + 1):
                            self.Add(self.pickup1[(k, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
                            self.Add(self.delivery1[(k, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
                    for d in range(self.D):
                        for t in range(-1, self.T + 1):
                            self.Add(self.pickup_d[(d, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
                            self.Add(self.delivery_d[(d, m, t)] == False).OnlyEnforceIf(self.req_cur_stage[(m, i)])
        
        # 配合上个条件，已经pickup的前一阶段必有更早的delivery    
        for t in range(-1, self.T + 1):  
            for m in range(self.M):
                for k in range(self.K):
                    prior_delivery_d = [self.delivery_d[(d, m, _t)] for _t in range(-1, t) for d in range(self.D)]
                    self.AddBoolOr(prior_delivery_d).OnlyEnforceIf(self.pickup2[(k, m, t)], self.req_cur_stage[(m, 1)].Not())
                for d in range(self.D):
                    prior_delivery1 = [self.delivery1[(k, m, _t)] for _t in range(-1, t) for k in range(self.K)]
                    self.AddBoolOr(prior_delivery1).OnlyEnforceIf(self.pickup_d[(d, m, t)], self.req_cur_stage[(m, 0)].Not())
        
    def _objective(self):
        # 最大化总价值
        total_value = 0
        for m in range(self.M):
            for k in range(self.K):
                for t in range(self.T + 1):
                    total_value += self.delivery2[(k, m, t)] * self.value[m]
        
        # 最小化代价
        for k in range(self.K):
            for t in range(self.T):
                cost_index = self.NewIntVar(0, self.N * self.N - 1, f'cost_index_k_{k}_{t}')
                self.Add(cost_index == self.location_k[(k, t)] * self.N + self.location_k[(k, t + 1)])
                cur_cost = self.NewIntVar(0, self.max_cost, f'cost_k_{k}_{t}')
                self.AddElement(cost_index, self.cost_K, cur_cost)
                total_value -= cur_cost
        
        for d in range(self.D):
            for t in range(self.T):
                cost_index = self.NewIntVar(0, self.N * self.N - 1, f'cost_index_d_{d}_{t}')
                self.Add(cost_index == self.location_d[(d, t)] * self.N + self.location_d[(d, t + 1)])
                cur_cost = self.NewIntVar(0, self.max_cost, f'cost_d_{d}_{t}')
                self.AddElement(cost_index, self.cost_D, cur_cost)
                total_value -= cur_cost

        self.Maximize(total_value)

    def solve(self, show=True):
        self._initialize_variables()
        self._add_constraints()
        self._objective()
        
        solver = cp_model.CpSolver()
        max_time_in_seconds = 200
        while True:
            solver.parameters.max_time_in_seconds = max_time_in_seconds
            solver.parameters.random_seed = 114514
            solver.parameters.num_workers = 4
            # solver.parameters.search_branching = cp_model.FIXED_SEARCH
            # solver.parameters.enumerate_all_solutions = False
            # solver.parameters.fix_variables_to_their_hinted_value = True
            # solver.parameters.min_num_lns_workers = 0
            # solver.parameters.log_search_progress = True
            status = solver.Solve(self)
            # 打印具体的状态码和名称，帮助诊断
            print(f"Solver returned status: {solver.StatusName(status)} ({status})")

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] or max_time_in_seconds >= 600:
                break
            else:
                max_time_in_seconds *= 2
                print(f"FEASIBLE solution haven't found, try to search {max_time_in_seconds} seconds.")

        # assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE], "怎么可能无解呢"
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("无解!!")
        _pickup1 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup2 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup_d = np.zeros((self.D, self.M, self.T + 1), dtype=bool)
        _location_k = np.full((self.K, self.T + 1), -1, dtype=int)
        _location_k[np.arange(self.K), self.join_time_K] = self.start_K
        _location_d = np.full((self.D, self.T + 1), -1, dtype=int)
        _location_d[np.arange(self.D), self.join_time_D] = self.start_D
    

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for m in range(self.M):
                for k in range(self.K):
                    for t in range(0, self.T + 1):
                        _pickup1[k, m, t] = solver.Value(self.pickup1[(k, m, t)])
                        _pickup2[k, m, t] = solver.Value(self.pickup2[(k, m, t)])
                for d in range(self.D):
                    for t in range(0, self.T + 1):
                        _pickup_d[d, m, t] = solver.Value(self.pickup_d[(d, m, t)])
            
            for k in range(self.K):
                for t in range(self.T + 1):
                    if solver.Value(self.no_action_k[(k, t)]) == 0:
                        _location_k[k, t] = solver.Value(self.location_k[(k, t)])
            
            for d in range(self.D):
                for t in range(self.T + 1):
                    if solver.Value(self.no_action_d[(d, t)]) == 0:
                        _location_d[d, t] = solver.Value(self.location_d[(d, t)])
        else:
            # no_action
            for k in range(self.K):
                _location_k[self.join_time_K[k]] = self.start_K[k]
            for d in range(self.D):
                _location_d[self.join_time_D[d]] = self.start_D[d]
        # assert(False)
        if not show:
            return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)
    
        
        return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)
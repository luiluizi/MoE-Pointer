from ortools.sat.python import cp_model
import numpy as np

from .annealing import SimulatedAnnealing

class CPSAT(cp_model.CpModel):
    def __init__(self, N, M, K ,T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load, objective_scale=1., env_args=None):
        """
        :param N: 节点数
        :param M: 请求数
        :param K: 车辆数
        :param T: 时间帧数
        :param start: 每辆车的起点
        :param capacity: 每辆车的容量
        :param join_time: 每辆车开始工作的时间，use for vehicles whose time_left != 0
        :param dist: 距离矩阵
        :param cost: 花费矩阵
        :param from_req: 每个请求的起点
        :param to_req: 每个请求的终点
        :param appear: 每个请求的出现时间
        :param value: 每个请求的价值
        :param penalty: 每个请求的惩罚
        :param pre_load: K x M，表示请求是否已经装载在了车上
        """
        super().__init__()

        # 常量定义
        if env_args is not None:
            self.max_dist = env_args["max_dist"]
            self.max_cost = round(env_args["max_dist"] * env_args["dist_cost"] / objective_scale)
        else:
            self.max_dist = 1000
            self.max_cost = 1000


        self.N = N
        self.M = M
        self.K = K
        self.T = T # [-1, T], 不过 T 时刻只有卸货，没有取货和下一跳。
        self.start = start
        self.capacity = capacity
        self.join_time = join_time
        self.ori_dist = dist
        self.dist = [self.NewIntVar(0, self.max_dist, f'dist_{i}_{j}') for i in range(self.N) for j in range(self.N)]
        self.cost = [self.NewIntVar(0, self.max_cost, f'cost_{i}_{j}') for i in range(self.N) for j in range(self.N)]
        self.from_req = from_req
        self.to_req = to_req
        self.appear = appear
        self.value = value
        self.penalty = penalty
        self.pre_load = pre_load
        self.objective_scale = objective_scale

        self._dist = dist
        self._cost = cost

        assert max(join_time) <= T
        if len(pre_load) != 0 and len(pre_load[0]) != 0:
            assert np.array(pre_load).sum(axis=0).max() <= 1, "同一个货物不可能提前被装在到两辆车上"
        assert (np.array(appear)[np.array(pre_load).any(axis=0)] == -1).all(), "提前装载的货物其 appear 需为 -1"
        for m in range(self.M):
            for k in range(self.K):
                assert not (self.pre_load[k][m] and self.join_time[k] == 0 and self.start[k] == to_req[m]), \
                    "不可能 t=0 就 delivery 货物"

        # 输入变量约束
        # dist[i][j] > 0
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    assert dist[i][j] == 0
                else:
                    assert dist[i][j] > 0

        # from_req[i] != to_req[i]
        for i in range(self.M):
            if self.from_req[i] == self.to_req[i]:
                raise ValueError(f'from_req[{i}] == to_req[{i}]')

        # dist约束为给定值
        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.dist, dist[i][j])

        for i in range(self.N):
            for j in range(self.N):
                self.AddElement(i * self.N + j, self.cost, cost[i][j])

        # 初始化决策变量
        self.pickup = {}
        self.delivery = {}

        # 初始化中间变量
        self.load = {}
        self.location = {}
        self.no_action = {}

        self.sa_solver = SimulatedAnnealing(N, M, K, T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load)
        self._initialize_variables()

    def _initialize_variables(self):
        _pickup, _location = self.sa_solver.solve(show=False, max_iter=10)

        # self.AddHint = lambda var, value: self.Add(var == value)

        for k in range(self.K):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.pickup[(k, m, t)] = self.NewBoolVar(f'pickup_{k}_{m}_{t}') # pickup[(*, *, T)] 没用
                    self.delivery[(k, m, t)] = self.NewBoolVar(f'delivery_{k}_{m}_{t}') # delivery[(*, *, -1/0)] 没用

                    # Hint
                    if t >= 0:
                        if m in self.sa_solver.best_solution[k][t]["pickup"] and not self.sa_solver.pre_load[k][m]:
                            self.AddHint(self.pickup[(k, m, t)], True)
                        else:
                            self.AddHint(self.pickup[(k, m, t)], False)
                        if m in self.sa_solver.best_solution[k][t]["delivery"]:
                            self.AddHint(self.delivery[(k, m, t)], True)
                        else:
                            self.AddHint(self.delivery[(k, m, t)], False)

        for k in range(self.K):
            for t in range(self.T + 1):
                self.load[(k, t)] = self.NewIntVar(0, self.capacity[k], f'load_{k}_{t}')
                self.location[(k, t)] = self.NewIntVar(-1, self.N - 1, f'location_{k}_{t}')
                self.no_action[(k, t)] = self.NewBoolVar(f'no_action_{k}_{t}') # no_cation[(*, T)] 没用

                # Hint
                self.AddHint(self.load[(k, t)], self.sa_solver.best_solution[k][t]["load"])
                if _location[k][t] != -1:
                    self.AddHint(self.location[(k, t)], _location[k][t])
                    self.AddHint(self.no_action[(k, t)], False)
                else:
                    self.AddHint(self.no_action[(k, t)], True)


    def _add_constraints(self):
        # 预设已经在车上的物品
        for k in range(len(self.pre_load)):
            for m in range(len(self.pre_load[0])):
                self.Add(self.pickup[(k, m, -1)] == self.pre_load[k][m])

        # 中间变量约束1： t > 0 -> load[k, t] = load[k, t-1] + sum(pickup[k, m, t]) - sum(delivery[k, m, t]); t = 0 -> load[k, t] = sum(pickup[k, m, t]) - sum(delivery[k, m, t])
        for k in range(self.K):
            for t in range(self.T + 1):
                if t == 0:
                    self.Add(self.load[(k, t)] == sum(self.pickup[(k, m, _t)] for m in range(self.M) for _t in range(-1, 1))) # t <= 0 时一定没有 delivery
                else:
                    self.Add(self.load[(k, t)] == self.load[(k, t-1)] + sum(self.pickup[(k, m, t)] for m in range(self.M)) - sum(self.delivery[(k, m, t)] for m in range(self.M)))

        # 中间变量约束2： t = 0 -> location[k,t] = start[k]; t > 0 -> location[k, t] = to_req[m] if delivery[k, m, t] , location[k, t] = from_req[m] if pickup[k, m, t]
        for k in range(self.K):
            self.Add(self.location[(k, 0)] == self.start[k])
            self.Add(self.no_action[(k, self.join_time[k])] == 0) # 车辆 join 时通过 no_action=False 来将其标记为关键点
            for t in range(self.T + 1):
                if t < self.join_time[k]:
                    self.Add(self.no_action[(k, t)] == 1) # 在车辆 join 之前需要保证车辆无操作
                if t != 0:
                    self.Add(self.location[(k, t)] == self.location[(k, t - 1)]).OnlyEnforceIf(self.no_action[(k, t - 1)])
                for m in range(self.M):
                    self.Add(self.location[(k, t)] == self.to_req[m]).OnlyEnforceIf(self.delivery[(k, m, t)])
                    self.Add(self.location[(k, t)] == self.from_req[m]).OnlyEnforceIf(self.pickup[(k, m, t)])

        # 中间变量约束3：sum(pickup[k, m, t]) + sum(delivery[k, m, t]) = 0 -> no_action[k, t]
        for k in range(self.K):
            for t in range(0, self.T + 1):
                if t == self.join_time[k]:
                    continue
                self.Add(self.no_action[(k, t)] == 1).OnlyEnforceIf(
                    [self.pickup[(k, m, t)].Not() for m in range(self.M)] +
                    [self.delivery[(k, m, t)].Not() for m in range(self.M)]
                )
                for m in range(self.M):
                    self.Add(self.no_action[(k, t)] == 0).OnlyEnforceIf(self.pickup[(k, m, t)])
                    self.Add(self.no_action[(k, t)] == 0).OnlyEnforceIf(self.delivery[(k, m, t)])

        # 约束1：每个请求只能被一辆车接送
        for m in range(self.M):
            self.Add(sum(self.pickup[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)
            self.Add(sum(self.delivery[(k, m, t)] for k in range(self.K) for t in range(-1, self.T + 1)) <= 1)

        # 约束2：若某个request被deliver， 则必须在之前的某个时间被同一辆车pickup
        for k in range(self.K):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.Add(sum(self.pickup[(k, m, _t)] for _t in range(-1, t)) == 1).OnlyEnforceIf(self.delivery[(k, m, t)])

        # 约束3：每个时刻每辆车的load不能超过capacity
        for k in range(self.K):
            for t in range(self.T + 1):
                self.Add(self.load[(k, t)] <= self.capacity[k])

        # 约束5：每辆车，任意两次操作的时间间隔应该大于两地的距离（第一种写法没考虑初始位置到第一次操作的距离，直接使用location枚举效率更高）
        for k in range(self.K):
            for t in range(self.T + 1):
                for t1 in range(t + 1, self.T + 1):
                    index = self.NewIntVar(0, self.N * self.N - 1, f'index_{k}_{t}_{t1}')
                    self.Add(index == self.location[(k, t)] * self.N + self.location[(k, t1)])
                    dis = self.NewIntVar(0, self.max_dist, f'dis_{k}_{t}_{t1}')
                    self.AddElement(index, self.dist, dis)
                    self.Add((t1 - t) >= dis).OnlyEnforceIf(self.no_action[(k, t)].Not(), self.no_action[(k, t1)].Not())

        # 约束6：pickup的时间应该在request出现之后
        for k in range(self.K):
            for m in range(self.M):
                for t in range(-1, self.T + 1):
                    self.Add(t >= self.appear[m]).OnlyEnforceIf(self.pickup[(k, m, t)])

    def _objective(self):
        # 最大化总价值
        total_value = 0
        for m in range(self.M):
            for k in range(self.K):
                for t in range(self.T + 1):
                    total_value += self.delivery[(k, m, t)] * self.value[m]

        
        # 最小化代价
        for k in range(self.K):
            for t in range(self.T):
                cost_index = self.NewIntVar(0, self.N * self.N - 1, f'cost_index_{k}_{t}')
                self.Add(cost_index == self.location[(k, t)] * self.N + self.location[(k, t + 1)])
                cur_cost = self.NewIntVar(0, self.max_cost, f'cost_{k}_{t}')
                self.AddElement(cost_index, self.cost, cur_cost)
                total_value -= cur_cost

        # 最小化惩罚
        # for m in range(self.M):
        #     served = self.NewBoolVar(f'served_{m}')
        #     for k in range(self.K):
        #         for t in range(self.T + 1):
        #             self.AddBoolOr([self.delivery[(k, m, t)] for k in range(self.K) for t in range(self.T + 1)]).OnlyEnforceIf(served)
        #             self.AddBoolOr([self.delivery[(k, m, t)].Not() for k in range(self.K) for t in range(self.T + 1)]).OnlyEnforceIf(served.Not())
        #     total_value -= served.Not() * self.penalty[m]

        self.Maximize(total_value)

    def solve(self, show=True):
        self._add_constraints()
        self._objective()

        solver = cp_model.CpSolver()
        max_time_in_seconds = 75
        while True:
            solver.parameters.max_time_in_seconds = max_time_in_seconds
            solver.parameters.random_seed = 114514
            solver.parameters.num_workers = 3
            # solver.parameters.fix_variables_to_their_hinted_value = True
            # solver.parameters.min_num_lns_workers = 0
            # solver.parameters.log_search_progress = True
            status = solver.Solve(self)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] or max_time_in_seconds >= 600:
                break
            else:
                max_time_in_seconds *= 2
                print(f"FEASIBLE solution haven't found, try to search {max_time_in_seconds} seconds.")

        # assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE], "怎么可能无解呢"
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("无解!!")

        _pickup = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _location = np.full((self.K, self.T + 1), -1, dtype=int)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for m in range(self.M):
                for k in range(self.K):
                    for t in range(0, self.T + 1):
                        _pickup[k, m, t] = solver.Value(self.pickup[(k, m, t)])
            
            for k in range(self.K):
                for t in range(self.T + 1):
                    if solver.Value(self.no_action[(k, t)]) == 0:
                        _location[k, t] = solver.Value(self.location[(k, t)])
        else:
            # no_action
            for k in range(self.K):
                _location[self.join_time[k]] = self.start[k]
        
        if not show:
            return _pickup, _location
    
        print('Status = %s' % solver.StatusName(status))
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Objective value = {solver.ObjectiveValue() * self.objective_scale}")
            print('================================================')
            for m in range(self.M):
                print(f'Request {m} from {self.from_req[m]} to {self.to_req[m]} at time {self.appear[m]}')
                for k in range(self.K):
                    for t in range(-1, self.T + 1):
                        if solver.Value(self.pickup[(k, m, t)]) == 1:
                            print(f'Car {k} picks up request {m} at t = {t}')
                        if solver.Value(self.delivery[(k, m, t)]) == 1:
                            print(f'Car {k} delivers request {m} at t = {t}')
                            assert t > 0, "t <= 0 不可能 delivery"
            print('================================================')
            for k in range(self.K):
                print(f'Car {k} route:')
                for t in range(self.T + 1):
                    print(f't = {t}, location = {solver.Value(self.location[(k, t)]), solver.Value(self.load[(k, t)])}')
        else:
            print('No solution found.')
        
        return _pickup, _location

if __name__ == '__main__':
    N = 5
    M = 4
    K = 2
    T = 10
    start = [0, 2]
    capacity = [2, 1]
    join_time = [2, 9]
    dist = [[0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]]
    cost = [[0] * N for _ in range(N)]
    from_req = [4, 2, 3, 3]
    to_req = [1, 4, 2, 0]
    appear = [0, 1, 3, 6]
    value = [dist[_from][to] for _from, to in zip(from_req, to_req)]
    penalty = [0] * len(from_req)
    pre_load = [[False] * M for _ in range(K)]
    pre_load[1][0] = True; appear[0] = -1;
    model = CPSAT(N, M, K, T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load)
    model.solve(show=True)
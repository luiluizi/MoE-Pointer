import math
import numpy as np
import copy


class GeneticAlgorithm:
    def __init__(self, N, M, K, T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load, **kwargs):
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
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.start = start
        self.capacity = capacity
        self.join_time = join_time
        self.dist = dist
        self.cost = cost
        self.from_req = from_req
        self.to_req = to_req
        self.appear = appear
        self.value = value
        self.penalty = penalty
        self.pre_load = pre_load

        self.rng = np.random.default_rng(114514)

    def initial_solution(self):
        solution = []
        for k in range(self.K):
            path = []
            for t in range(0, self.T + 1):
                path.append({'location': self.start[k], 'pickup': [], 'delivery': [], 'load': 0})
            solution.append(path)

        # 处理预装载的请求，记录为0时刻被车辆 k pickup
        for m in range(self.M):
            for k in range(self.K):
                if self.pre_load[k][m]:
                    solution[k][0]['pickup'].append(m)
        
        for k in range(self.K):
            # 更新车辆位置和容量
            self.update_vehicle_location(solution[k], k)
            self.update_vehicle_load(solution[k])

        if self.check_constraints(solution):
            return solution
        else:
            return []


    def update_vehicle_location(self, vehicle_path, vehicle_id):
        # 更新路径，保证无操作时刻的位置和紧邻上一有操作时刻的位置相同
        prev_location = self.start[vehicle_id]
        for t in range(1, self.T + 1):
            if vehicle_path[t]['pickup'] or vehicle_path[t]['delivery']:
                # 更新当前位置
                vehicle_path[t]['location'] = self.from_req[vehicle_path[t]['pickup'][0]] if vehicle_path[t]['pickup'] \
                    else self.to_req[vehicle_path[t]['delivery'][0]]
                prev_location = vehicle_path[t]['location']
            else:
                vehicle_path[t]['location'] = prev_location

    def update_vehicle_load(self, vehicle_path):
        # 更新车辆容量，初始为0，每个时刻增加pickup数量，减少delivery数量
        vehicle_path[0]['load'] = len(vehicle_path[0]['pickup']) - len(vehicle_path[0]['delivery'])
        for t in range(1, self.T + 1):
            vehicle_path[t]['load'] = vehicle_path[t - 1]['load'] + len(vehicle_path[t]['pickup']) - len(vehicle_path[t]['delivery'])

    def check_constraints(self, solution):
        # 检查初始位置约束
        for k in range(self.K):
            if solution[k][0]['location'] != self.start[k] or solution[k][self.join_time[k]]['location'] != self.start[k]:
                return False

        # 检查位移约束，相邻两个有行动的时间帧间的距离小于时间差（t = join_time[k]算作有行动）
        for k in range(self.K):
            last_t = None
            for t in range(self.T + 1):
                if (t == self.join_time[k] or solution[k][t]['pickup'] or solution[k][t]['delivery']):
                    if last_t is not None:
                        if self.dist[solution[k][last_t]['location']][solution[k][t]['location']] > t - last_t:
                            return False
                    last_t = t

        # 检查位置一致性约束，同一时刻所有操作的位置一致
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution[k][t]['pickup']:
                    # 跳过预装载的请求
                    if self.pre_load[k][m]:
                        continue
                    if solution[k][t]['location'] != self.from_req[m]:
                        return False
                for m in solution[k][t]['delivery']:
                    if solution[k][t]['location'] != self.to_req[m]:
                        return False

        # 检查请求唯一性
        pickup_count = [0] * self.M
        delivery_count = [0] * self.M
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution[k][t]['pickup']:
                    pickup_count[m] += 1
                for m in solution[k][t]['delivery']:
                    delivery_count[m] += 1
        for m in range(self.M):
            if pickup_count[m] > 1 or delivery_count[m] > 1:
                return False

        # 检查容量约束
        for k in range(self.K):
            for t in range(self.T + 1):
                if solution[k][t]['load'] < 0 or solution[k][t]['load'] > self.capacity[k]:
                    return False

        # 检查出现时间约束
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution[k][t]['pickup']:
                    if t < self.appear[m]:
                        return False

        # 检查车辆开始工作时间约束
        for k in range(self.K):
            for t in range(self.join_time[k]):
                if solution[k][t]['delivery']:
                    return False
                elif solution[k][t]['pickup']:
                    # 检查是不是预装载的请求，有不是的直接返回False
                    if t == 0:
                        for m in solution[k][t]['pickup']:
                            if not self.pre_load[k][m]:
                                return False
                    else:
                        return False

        return True

    def objective_function(self, solution):
        # 目标函数：总价值 - 总成本 - 总惩罚
        total_value = 0
        total_cost = 0
        unserved = set(range(self.M))

        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution[k][t]['delivery']:
                    # 计算总价值
                    total_value += self.value[m]
                    unserved.discard(m)
                if t > 0:
                    # 计算总成本
                    prev_location = solution[k][t - 1]['location']
                    curr_location = solution[k][t]['location']
                    total_cost += self.cost[prev_location][curr_location]

        # 未完成的请求的总惩罚
        total_penalty = sum(self.penalty[m] for m in unserved)
        return math.exp((total_value - total_cost - total_penalty) / max(sum(self.value), 1) * 10) # 适应度函数，保证为正数 # 改为超参

    def crossover(self, parent1, parent2):
        # 交叉操作：随机选择一个请求，交换两个父代中该请求的分配
        retries = 0
        while retries < 5:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            if self.M == 0:
                break
            m = self.rng.integers(0, self.M)

            # 如果是预装载请求，只交换 delivery 时刻
            if any(self.pre_load[k][m] for k in range(self.K)):
                # 找到预装载请求的车辆
                k = next(k for k in range(self.K) if self.pre_load[k][m])

                # 移除该请求的旧 delivery 分配
                for t in range(self.T + 1):
                    if m in child1[k][t]['delivery']:
                        child1[k][t]['delivery'].remove(m)
                    if m in child2[k][t]['delivery']:
                        child2[k][t]['delivery'].remove(m)

                # 从父代中复制该请求的 delivery 分配
                for t in range(self.T + 1):
                    if m in parent2[k][t]['delivery']:
                        child1[k][t]['delivery'].append(m)
                    if m in parent1[k][t]['delivery']:
                        child2[k][t]['delivery'].append(m)
            else:
                # 移除该请求的旧分配
                # 非预装载请求，需要交换 pickup 和 delivery 时刻
                for k in range(self.K):
                    for t in range(self.T + 1):
                        if m in child1[k][t]['pickup']:
                            child1[k][t]['pickup'].remove(m)
                        if m in child1[k][t]['delivery']:
                            child1[k][t]['delivery'].remove(m)
                        if m in child2[k][t]['pickup']:
                            child2[k][t]['pickup'].remove(m)
                        if m in child2[k][t]['delivery']:
                            child2[k][t]['delivery'].remove(m)

                # 从父代中复制该请求的分配
                for k in range(self.K):
                    for t in range(self.T + 1):
                        if m in parent2[k][t]['pickup']:
                            child1[k][t]['pickup'].append(m)
                        if m in parent2[k][t]['delivery']:
                            child1[k][t]['delivery'].append(m)
                        if m in parent1[k][t]['pickup']:
                            child2[k][t]['pickup'].append(m)
                        if m in parent1[k][t]['delivery']:
                            child2[k][t]['delivery'].append(m)

                # 更新车辆位置和容量
                for k in range(self.K):
                    self.update_vehicle_location(child1[k], k)
                    self.update_vehicle_load(child1[k])
                    self.update_vehicle_location(child2[k], k)
                    self.update_vehicle_load(child2[k])

            if self.check_constraints(child1) and self.check_constraints(child2):
                return child1, child2

            retries += 1

        # 未找到合法解，返回原父代
        return parent1, parent2

    def mutate(self, solution):
        # 变异操作：随机选择一个请求，重新分配其车辆和时刻
        retries = 0
        while retries < 5:
            new_solution = copy.deepcopy(solution)

            if self.M == 0:
                break
            # 随机选择一个请求
            m = self.rng.integers(0, self.M)
            old_k = -1

            # 移除该请求的旧分配
            for k in range(self.K):
                for t in range(self.T + 1):
                    if m in new_solution[k][t]['pickup']:
                        old_k = k
                        new_solution[k][t]['pickup'].remove(m)
                    if m in new_solution[k][t]['delivery']:
                        old_k = k
                        new_solution[k][t]['delivery'].remove(m)

            # 按概率重新分配或不分配请求
            if not self.pre_load[old_k][m] and self.rng.random() < math.exp(-1 / ( retries + 1)):
                return new_solution

            # 如果是预装载请求，只改变delivery时刻；否则重新分配车辆，pickup和delivery时刻
            if self.pre_load[old_k][m]:
                k = old_k
                new_solution[k][0]['pickup'].append(m)
                if self.dist[self.start[k]][self.to_req[m]] + self.join_time[k] <= self.T:
                    t_delivery = self.rng.integers(self.dist[self.start[k]][self.to_req[m]] + self.join_time[k], self.T + 1)
                    new_solution[k][t_delivery]['delivery'].append(m)
            else:
                k = self.rng.integers(0, self.K)
                t_pickup = self.rng.integers(max(self.join_time[k], self.appear[m]), self.T + 1)
                new_solution[k][t_pickup]['pickup'].append(m)
                if t_pickup + self.dist[self.from_req[m]][self.to_req[m]] <= self.T:
                    t_delivery = self.rng.integers(t_pickup + self.dist[self.from_req[m]][self.to_req[m]], self.T + 1)
                    new_solution[k][t_delivery]['delivery'].append(m)
            
            for k in range(self.K):
                # 更新车辆位置和容量
                self.update_vehicle_location(new_solution[k], k)
                self.update_vehicle_load(new_solution[k])

            if self.check_constraints(new_solution):
                return new_solution

            retries += 1

        # 未找到合法解，返回原解
        return solution

    def solve(self, population_size=10, mutation_rate=0.3, crossover_rate=0.5, max_generations=1000, show=True):
        # 初始化种群
        population = [self.initial_solution() for _ in range(population_size)]
        if not all(population):
            return [], -1

        # 计算初始种群的适应度
        fitness = [self.objective_function(ind) for ind in population]

        # 记录最佳解
        best_solution = population[np.argmax(fitness)]
        best_fitness = max(fitness)

        # 迭代进化
        for generation in range(max_generations):
            new_population = []

            # 选择、交叉、变异
            for _ in range(population_size // 2):
                # 选择
                normlized_fitness = np.array(fitness).clip(min=1e-5)
                normlized_fitness = normlized_fitness / normlized_fitness.sum()
                parent1_id, parent2_id = self.rng.choice(list(range(len(population))), p=normlized_fitness, size=2)
                parent1, parent2 = population[parent1_id], population[parent2_id]

                # 交叉
                if self.rng.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # 变异
                if self.rng.random() < mutation_rate:
                    child1 = self.mutate(child1)
                if self.rng.random() < mutation_rate:
                    child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            # 更新种群
            population = new_population
            fitness = [self.objective_function(ind) for ind in population]

            # 更新最佳解
            current_best_fitness = max(fitness)
            if current_best_fitness > best_fitness:
                best_solution = population[np.argmax(fitness)]
                best_fitness = current_best_fitness

            # print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
        
        assert best_solution is not None, "No feasible solution found."

        _pickup = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _location = np.full((self.K, self.T + 1), -1, dtype=int)
        _location[np.arange(self.K), self.join_time] = self.start
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in best_solution[k][t]['pickup']:
                    if not self.pre_load[k][m]:
                        _pickup[k, m, t] = True
                        assert _location[k, t] == -1 or _location[k, t] == self.from_req[m]
                        _location[k, t] = self.from_req[m]

                for m in best_solution[k][t]['delivery']:
                    assert _location[k, t] == -1 or _location[k, t] == self.to_req[m]
                    _location[k, t] = self.to_req[m]

        if not show:
            return _pickup, _location
        
        print("Best Objective Value:", int(math.log(best_fitness, math.e))) # 适应度函数为 e^x，取对数为 e 为目标函数值
        print("=======================Request Result=======================")
        for m in range(self.M):
            print(f"Request {self.m} From: {self.from_req[m]}, To: {self.to_req[m]}, Appear: {self.appear[m]}, Value: {self.value[m]}, Penalty: {self.penalty[m]}")
            for k in range(self.K):
                for t in range(self.T + 1):
                    if m in best_solution[k][t]['pickup']:
                        print(f"Car {k} Pickup Request {m} at t = {t}")
                    if m in best_solution[k][t]['delivery']:
                        print(f"Car {k} Delivery Request {m} at t = {t}")
        print("=======================Vehicle Result=======================")
        for k in range(self.K):
            print(f"Car {k} Route:")
            for t in range(self.T + 1):
                print(
                    f"t = {t}, location = {best_solution[k][t]['location']}, load = {best_solution[k][t]['load']}, pickup = {best_solution[k][t]['pickup']}, delivery = {best_solution[k][t]['delivery']}")
        
        return _pickup, _location
    
if __name__ == "__main__":
    # 示例数据
    N = 5
    M = 6
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
    from_req = [4, 2, 3, 3, 1, 0]
    to_req = [1, 4, 2, 0, 4, 3]
    appear = [0, 1, 3, 6, 8, 9]
    cost = [[0] * N for _ in range(N)]
    value = [dist[_from][to] for _from, to in zip(from_req, to_req)]
    penalty = [0] * len(from_req)
    pre_load = [[False] * M for _ in range(K)]
    pre_load[1][0] = True
    appear[0] = -1

    # 初始化并运行遗传算法
    ga = GeneticAlgorithm(N, M, K, T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load)
    ga.solve()
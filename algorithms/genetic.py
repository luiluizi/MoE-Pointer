import math
import numpy as np
import copy
from .component.metaheuristic import MetaheuristicBase


class GeneticAlgorithm(MetaheuristicBase):
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=3.0, courier_stage1_temp=None, **kwargs):
        """初始化遗传算法，调用基类初始化方法"""
        super().__init__(N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio, courier_stage1_temp, **kwargs)
        self.couriers = []
        self.drones = []
        self.requests = []
        self.change_item = ["pickup1", "delivery1", "pickup", "delivery", "pickup2", "delivery2"]

    def initial_solution(self):
        """生成初始解，调用基类方法"""
        return super().initial_solution()

    def objective_function(self, solution):
        solution_K, solution_D = solution
        total_value = 0
        total_cost = 0
        unserved = set(range(self.M))
        
        stage1_reward = 0.15
        stage2_reward = 0.15   
        stage3_reward = 1.0   
        
        # 记录每个请求完成的阶段
        request_stage1_done = set()
        request_stage2_done = set()
        request_stage3_done = set()
        
        # 统计Courier的Stage1和Stage3收益
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['delivery1']:
                    if m not in request_stage1_done:
                        total_value += stage1_reward * self.value[m]
                        request_stage1_done.add(m)
                for m in solution_K[k][t]['delivery2']:
                    if m not in request_stage3_done:
                        total_value += stage3_reward * self.value[m]
                        request_stage3_done.add(m)
                        unserved.discard(m)
                if t > 0:
                    prev_location = solution_K[k][t - 1]['location']
                    curr_location = solution_K[k][t]['location']
                    total_cost += self.cost_K[prev_location][curr_location]

        # 统计Drone的Stage2收益
        for d in range(self.D):
            for t in range(self.T + 1):
                for m in solution_D[d][t]['delivery']:
                    if m not in request_stage2_done:
                        total_value += stage2_reward * self.value[m]
                        request_stage2_done.add(m)
                if t > 0:
                    prev_location = solution_D[d][t - 1]['location']
                    curr_location = solution_D[d][t]['location']
                    total_cost += self.cost_D[prev_location][curr_location]
        
        # 未完成的请求的总惩罚（只有未完成Stage3的请求才计入惩罚）
        total_penalty = sum(self.penalty[m] for m in unserved)
        # 直接返回目标函数值（与模拟退火一致）
        return total_value - total_cost - total_penalty

    def exchange(self, parent1, parent2, child1, child2, m, stage_id, pre_assign=-1):
        self.exchange_one_stage(parent1, parent2, child1, child2, m, stage_id, pre_assign)
        for i in range(stage_id + 1, 6):
            self.exchange_one_stage(parent1, parent2, child1, child2, m, stage_id=i, pre_assign=-1)
    
    def exchange_one_stage(self, parent1, parent2, child1, child2, m, stage_id, pre_assign):
        item = self.change_item[stage_id]
        if stage_id in [0, 1, 4, 5]:
            solution1_C, solution2_C, solution1_P, solution2_P = child1[0], child2[0], parent1[0], parent2[0]
            num_vehicles = self.K
        elif stage_id in [2, 3]:
            solution1_C, solution2_C, solution1_P, solution2_P = child1[1], child2[1], parent1[1], parent2[1]
            num_vehicles = self.D
        if pre_assign != -1:
            k = pre_assign
            for t in range(self.T + 1):
                if m in solution1_C[k][t][item]:
                    solution1_C[k][t][item].remove(m)
                if m in solution2_C[k][t][item]:
                    solution2_C[k][t][item].remove(m)
            for t in range(self.T + 1):
                if m in solution2_P[k][t][item]:
                    solution1_C[k][t][item].append(m)
                if m in solution1_P[k][t][item]:
                    solution2_C[k][t][item].append(m)
        else:
            for k in range(num_vehicles):
                for t in range(self.T + 1):
                    if m in solution1_C[k][t][item]:
                        solution1_C[k][t][item].remove(m)
                    if m in solution2_C[k][t][item]:
                        solution2_C[k][t][item].remove(m)
            # 从父代中复制该请求的分配
            for k in range(num_vehicles):
                for t in range(self.T + 1):
                    if m in solution2_P[k][t][item]:
                        solution1_C[k][t][item].append(m)
                    if m in solution1_P[k][t][item]:
                        solution2_C[k][t][item].append(m)

    def crossover(self, parent1, parent2):
        # 交叉操作：随机选择一个请求，交换两个父代中该请求的分配
        retries = 0
        while retries < 5:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            if self.M == 0:
                break
            m = self.rng.integers(0, self.M)

            if any(self.pre_load_K_stage1[k][m] for k in range(self.K)):
                k = next(k for k in range(self.K) if self.pre_load_K_stage1[k][m])
                self.exchange(parent1, parent2, child1, child2, m, 1, k)
            elif any(self.pre_load_D[d][m] for d in range(self.D)):
                d = next(d for d in range(self.D) if self.pre_load_D[d][m])
                self.exchange(parent1, parent2, child1, child2, m, 3, d)
            elif any(self.pre_load_K_stage3[k][m] for k in range(self.K)):
                k = next(k for k in range(self.K) if self.pre_load_K_stage3[k][m])
                self.exchange(parent1, parent2, child1, child2, m, 5, k)
            elif self.wait_stage2[m]:
                self.exchange(parent1, parent2, child1, child2, m, 2)
            elif self.wait_stage3[m]:
                self.exchange(parent1, parent2, child1, child2, m, 4)
            else:
                self.exchange(parent1, parent2, child1, child2, m, 0)
            # 更新位置和容量
            self._update_all_vehicles(child1[0], child1[1])
            self._update_all_vehicles(child2[0], child2[1])

            if self.check_constraints(child1[0], child1[1]) and self.check_constraints(child2[0], child2[1]):
                return child1, child2

            retries += 1

        # 未找到合法解，返回原父代
        return parent1, parent2

    def mutate(self, solution):
        solution_K, solution_D = solution
        retries = 0
        while retries < 5:
            new_solution_K = copy.deepcopy(solution_K)
            new_solution_D = copy.deepcopy(solution_D)
            # 随机选择一个请求
            if self.M == 0:
                break
            m = self.rng.integers(0, self.M)
            old_k1 = -1
            old_k2 = -1
            old_d = -1
            
            # TODO 实现1 每次清空三阶段的全部操作，重新进行分配 实现2 按照概率 要不清空三阶段全部操作，要不只清空某阶段操作
            old_k1, old_d, old_k2 = self.find_old_id(new_solution_K, new_solution_D, m)
            # 按概率重新分配或不分配请求
            # 不能在任一阶段有预分配
            # if (not ((old_k1 != -1 and self.pre_load_K_stage1[old_k1][m]) or (old_k2 != -1 and self.pre_load_K_stage3[old_k2][m]) or (old_d != -1 and self.pre_load_D[old_d][m]))) and self.rng.random() < math.exp(-1 / ( retries + 1)):
            #     self.remove_stage1(new_solution_K, new_solution_D, m)
            #     self.remove_stage2(new_solution_K, new_solution_D, m)
            #     self.remove_stage3(new_solution_K, new_solution_D, m)
            #     return new_solution_K, new_solution_D
            
            p1 = self.rng.random()
            if p1 < 0.2:
                end_time = self.find_pickup_d_time(solution_D, m)
                if end_time is None:
                    end_time = self.T
                self.reassign_stage1(new_solution_K, new_solution_D, m, end_time)
            elif p1 < 0.6:
                # 查看阶段3开始时间
                end_time = self.find_pickup2_time(solution_K, m)
                if end_time is None:
                    end_time = self.T
                self.reassign_stage2(new_solution_K, new_solution_D, m, end_time)
            else: 
                self.reassign_stage3(new_solution_K, new_solution_D, m, self.T)
                
                # 更新车辆位置和容量
            self._update_all_vehicles(new_solution_K, new_solution_D)
            # 检查新解是否满足约束
            if self.check_constraints(new_solution_K, new_solution_D):
                return new_solution_K, new_solution_D

            retries += 1

        # 未找到合法解，返回原解
        return solution_K, solution_D
    

    def solve(self, population_size=10, mutation_rate=0.3, crossover_rate=0.5, max_generations=1000, show=True):
    # def solve(self, population_size=10, mutation_rate=0.4, crossover_rate=0.5, max_generations=2000, show=True):
        # 初始化种群
        population = [self.initial_solution() for _ in range(population_size)]
        if not all(population):
            return [], -1

        # 计算初始种群的适应度
        fitness = [self.objective_function(ind) for ind in population]
        # 记录最佳解
        best_solution = population[np.argmax(fitness)]
        best_fitness = max(fitness)

        for generation in range(max_generations):
            elite = copy.deepcopy(best_solution)
            
            new_population = []
            
            for _ in range(population_size // 2):
                fitness_array = np.array(fitness)
                min_fitness = fitness_array.min()
                if min_fitness < 0:
                    fitness_array = fitness_array - min_fitness + 1e-5
                else:
                    fitness_array = fitness_array + 1e-5

                normalized_fitness = fitness_array / fitness_array.sum()
                parent1_id, parent2_id = self.rng.choice(list(range(len(population))), p=normalized_fitness, size=2)
                parent1, parent2 = population[parent1_id], population[parent2_id]

                # 交叉
                if self.rng.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                if self.rng.random() < mutation_rate:
                    child1 = self.mutate(child1)
                if self.rng.random() < mutation_rate:
                    child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            # 更新种群
            population = new_population
            fitness = [self.objective_function(ind) for ind in population]
            
            # 精英保留：用精英个体替换最差个体
            worst_idx = np.argmin(fitness)
            population[worst_idx] = elite
            fitness[worst_idx] = best_fitness
            
            # 更新最佳解
            current_best_fitness = max(fitness)
            if current_best_fitness > best_fitness:
                best_solution = population[np.argmax(fitness)]
                best_fitness = current_best_fitness
            else:
                # 如果新种群中没有更好的解，确保best_solution仍然是最佳的
                best_solution = elite
            
        best_solution_K, best_solution_D = best_solution
        assert best_solution_K is not None and best_solution_D is not None, "No feasible solution found."
        
        _pickup1, _pickup2, _pickup_d, _location_k, _location_d = self._convert_solution_to_output(best_solution_K, best_solution_D)

        if not show:
            return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)
        
        self._print_solution(best_solution_K, best_solution_D, best_fitness, is_genetic=True)
        return _pickup1, _pickup2, _pickup_d, _location_k, _location_d
    
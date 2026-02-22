import math
import copy
from .component.metaheuristic import MetaheuristicBase

class SimulatedAnnealing(MetaheuristicBase):
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=3.0, courier_stage1_temp=None, **kwargs):
        super().__init__(N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio, courier_stage1_temp, **kwargs)

    def initial_solution(self):
        solution_K, solution_D = super().initial_solution()
        if not solution_K and not solution_D:
            self._check_constraints(solution_K, solution_D)
            assert(False)
        return solution_K, solution_D

    def neighbor_solution(self, solution_K, solution_D):
        # 随机选择一个邻域操作
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

    def objective_function(self, solution_K, solution_D):
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
        return total_value - total_cost - total_penalty

    def solve(self, initial_temp=1000, final_temp=1, cooling_rate=0.99, max_iter=5000, show=True):
        current_solution_K, current_solution_D = self.initial_solution()
        if not current_solution_K and not current_solution_D:
            print("No feasible solution found.")
            return [], [], [], [-1], [-1]

        current_obj = self.objective_function(current_solution_K, current_solution_D)
        best_solution_K, best_solution_D = current_solution_K, current_solution_D
        best_obj = current_obj

        temp = initial_temp
        for i in range(max_iter):
            # if temp < final_temp:
            #     break
            neighbor_K, neighbor_D = self.neighbor_solution(current_solution_K, current_solution_D)
            neighbor_obj = self.objective_function(neighbor_K, neighbor_D)

            if neighbor_obj > current_obj or self.rng.random() < math.exp((neighbor_obj - current_obj) / temp):
                current_solution_K = neighbor_K
                current_solution_D = neighbor_D
                current_obj = neighbor_obj

            if current_obj > best_obj:
                best_solution_K = current_solution_K
                best_solution_D = current_solution_D
                best_obj = current_obj

            temp *= cooling_rate
        
        self.best_solution_K = best_solution_K
        self.best_solution_D = best_solution_D
        assert best_solution_K is not None and best_solution_D is not None, "No feasible solution found."

        # 使用基类方法转换解为输出格式
        _pickup1, _pickup2, _pickup_d, _location_k, _location_d = self._convert_solution_to_output(best_solution_K, best_solution_D)
        
        if not show:
            return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)

        # 使用基类方法打印结果
        self._print_solution(best_solution_K, best_solution_D, best_obj, is_genetic=False)
        return _pickup1, _pickup2, _pickup_d, _location_k, _location_d
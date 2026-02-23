import math
import copy
from .component.metaheuristic import MetaheuristicBase

class SimulatedAnnealing(MetaheuristicBase):
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, 
                 cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, 
                 pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=3.0, courier_stage1_temp=None, **kwargs):
        super().__init__(N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, 
                         cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, 
                         pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio, courier_stage1_temp, **kwargs)

    def initial_solution(self):
        solution_K, solution_D = super().initial_solution()
        if not solution_K and not solution_D:
            self._check_constraints(solution_K, solution_D)
            assert(False)
        return solution_K, solution_D

    def neighbor_solution(self, solution_K, solution_D):
        # Randomly select a neighborhood operation
        retries = 0
        while retries < 5:
            new_solution_K = copy.deepcopy(solution_K)
            new_solution_D = copy.deepcopy(solution_D)
            # Randomly select a request
            if self.M == 0:
                break
            m = self.rng.integers(0, self.M)
            old_k1 = -1
            old_k2 = -1
            old_d = -1
            
            old_k1, old_d, old_k2 = self.find_old_id(new_solution_K, new_solution_D, m)
            # Reassign or unassign the request probabilistically
            # No pre-assignment allowed in any stage
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
                # Check start time of Stage 3
                end_time = self.find_pickup2_time(solution_K, m)
                if end_time is None:
                    end_time = self.T
                self.reassign_stage2(new_solution_K, new_solution_D, m, end_time)
            else: 
                self.reassign_stage3(new_solution_K, new_solution_D, m, self.T)
                
            # Update vehicle positions and capacities
            self._update_all_vehicles(new_solution_K, new_solution_D)
            # Check if the new solution satisfies all constraints
            if self.check_constraints(new_solution_K, new_solution_D):
                return new_solution_K, new_solution_D
            retries += 1
        # No valid solution found, return the original solution
        return solution_K, solution_D

    def objective_function(self, solution_K, solution_D):
        total_value = 0
        total_cost = 0
        unserved = set(range(self.M))
        
        stage1_reward = 0.15
        stage2_reward = 0.15   
        stage3_reward = 1.0   
        
        # Record the completed stages for each request
        request_stage1_done = set()
        request_stage2_done = set()
        request_stage3_done = set()
        
        # Calculate rewards for Courier's Stage 1 and Stage 3
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

        # Calculate rewards for Drone's Stage 2
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
        
        # Total penalty for unserved requests (only requests with incomplete Stage 3 are counted)
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

        # Use base class method to convert solution to output format
        self._print_solution(best_solution_K, best_solution_D, best_obj, is_genetic=False)
        return _pickup1, _pickup2, _pickup_d, _location_k, _location_d
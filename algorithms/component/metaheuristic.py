"""
元启发式算法通用组件基类
包含 SimulatedAnnealing 和 GeneticAlgorithm 的通用方法
"""
import numpy as np
import math


class MetaheuristicBase:
    """
    元启发式算法基类
    包含所有通用方法，供 SimulatedAnnealing 和 GeneticAlgorithm 继承
    """
    
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=3.0, courier_stage1_temp=None, **kwargs):
        """
        :param N: Number of nodes
        :param M: Number of requests/orders
        :param K: Number of couriers
        :param D: Number of drones
        :param T: Number of time frames
        :param start_K: Starting point of each courier
        :param start_D: Starting point of each drone
        :param capacity: Capacity of each vehicle
        :param join_time_K: Start working time of each courier, used for vehicles with time_left != 0
        :param join_time_D: Start working time of each drone, used for vehicles with time_left != 0
        :param dist: Distance matrix
        :param cost_K: Cost matrix for couriers
        :param cost_D: Cost matrix for drones
        :param from_req: Starting point of each request
        :param to_req: End point of each request
        :param station1_req: Station 1 for each request
        :param station2_req: Station 2 for each request
        :param appear: Appearance time of each request
        :param value: Value of each request
        :param penalty: Penalty for each request
        :param pre_load_K_stage1: K x M matrix, indicating whether a request has been loaded onto a courier in stage 1
        :param pre_load_K_stage3: K x M matrix, indicating whether a request has been loaded onto a courier in stage 3
        :param pre_load_D: D x M matrix, indicating whether a request has been loaded onto a drone
        :param wait_stage2: M-dimensional array, indicating whether a request is waiting to enter stage 2
        :param wait_stage3: M-dimensional array, indicating whether a request is waiting to enter stage 3
        :param courier_stage1_temp: Records the courier that completed stage 1 delivery for each order, to prevent the same courier from being used for both stage 1 and 3
        """
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
        self.cost_K = cost_K
        self.cost_D = cost_D
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
        self.req_cur_stage = [0] * self.M
        self.courier_stage1_temp = courier_stage1_temp
        
        self.rng = np.random.default_rng(120)
    
    def update_courier_location(self, courier_path, courier_id):
        """Update paths to ensure that the position during idle time is 
        the same as the position at the immediately preceding active time step"""
        prev_location = self.start_K[courier_id]
        for t in range(1, self.T + 1):
            if courier_path[t]['pickup1']:
                courier_path[t]['location'] = self.from_req[courier_path[t]['pickup1'][0]]
                prev_location = courier_path[t]['location']
            elif courier_path[t]['delivery1']:
                courier_path[t]['location'] = self.station1_req[courier_path[t]['delivery1'][0]]
                prev_location = courier_path[t]['location']
            elif courier_path[t]['pickup2']:
                courier_path[t]['location'] = self.station2_req[courier_path[t]['pickup2'][0]]
                prev_location = courier_path[t]['location']
            elif courier_path[t]['delivery2']:
                courier_path[t]['location'] = self.to_req[courier_path[t]['delivery2'][0]]
                prev_location = courier_path[t]['location']
            else:
                courier_path[t]['location'] = prev_location
    
    def update_drone_location(self, drone_path, drone_id):
        """Update paths to ensure that the position during idle time is 
        the same as the position at the immediately preceding active time step"""
        prev_location = self.start_D[drone_id]
        for t in range(1, self.T + 1):
            if drone_path[t]['pickup'] or drone_path[t]['delivery']:
                # 更新当前位置
                drone_path[t]['location'] = self.station1_req[drone_path[t]['pickup'][0]] if drone_path[t]['pickup'] \
                    else self.station2_req[drone_path[t]['delivery'][0]]
                prev_location = drone_path[t]['location']
            else:
                drone_path[t]['location'] = prev_location

    def update_courier_load(self, courier_path):
        """Update the load of couriers"""
        courier_path[0]['load'] = len(courier_path[0]['pickup1']) - len(courier_path[0]['delivery1']) + len(courier_path[0]['pickup2']) - len(courier_path[0]['delivery2']) 
        for t in range(1, self.T + 1):
            courier_path[t]['load'] = courier_path[t - 1]['load'] + len(courier_path[t]['pickup1']) - len(courier_path[t]['delivery1']) + len(courier_path[t]['pickup2']) - len(courier_path[t]['delivery2']) 
    
    def update_drone_load(self, drone_path):
        """Update the load of drones"""
        drone_path[0]['load'] = len(drone_path[0]['pickup']) - len(drone_path[0]['delivery'])
        for t in range(1, self.T + 1):
            drone_path[t]['load'] = drone_path[t - 1]['load'] + len(drone_path[t]['pickup']) - len(drone_path[t]['delivery'])
    
    def _update_all_vehicles(self, solution_K, solution_D):
        """Update the positions and loads of all vehicles"""
        for k in range(self.K):
            self.update_courier_location(solution_K[k], k)
            self.update_courier_load(solution_K[k])
        for d in range(self.D):
            self.update_drone_location(solution_D[d], d)
            self.update_drone_load(solution_D[d])
    
    def check_constraints(self, solution_K, solution_D):
        """
        Check if the solution satisfies all constraints
        Returns True if all constraints are satisfied, otherwise returns False
        """
        # 1. Check capacity constraints
        for k in range(self.K):
            for t in range(self.T + 1):
                if solution_K[k][t]['load'] < 0 or solution_K[k][t]['load'] > self.capacity[k]:
                    return False

        for d in range(self.D):
            for t in range(self.T + 1):
                if solution_D[d][t]['load'] < 0 or solution_D[d][t]['load'] > 1:
                    return False
        
        # 2. Check initial position constraints
        for k in range(self.K):
            if solution_K[k][0]['location'] != self.start_K[k] or solution_K[k][self.join_time_K[k]]['location'] != self.start_K[k]:
                return False
        for d in range(self.D):
            if solution_D[d][0]['location'] != self.start_D[d] or solution_D[d][self.join_time_D[d]]['location'] != self.start_D[d]:
                return False

        # 3. Check request uniqueness
        pickup1_count = [0] * self.M
        delivery1_count = [0] * self.M
        pickup2_count = [0] * self.M
        delivery2_count = [0] * self.M
        pickup_d_count = [0] * self.M
        delivery_d_count = [0] * self.M
        
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['pickup1']:
                    pickup1_count[m] += 1
                    if pickup1_count[m] > 1:
                        return False
                for m in solution_K[k][t]['delivery1']:
                    delivery1_count[m] += 1
                    if delivery1_count[m] > 1:
                        return False
                for m in solution_K[k][t]['pickup2']:
                    pickup2_count[m] += 1
                    if pickup2_count[m] > 1:
                        return False
                for m in solution_K[k][t]['delivery2']:
                    delivery2_count[m] += 1
                    if delivery2_count[m] > 1:
                        return False
        
        for d in range(self.D):
            for t in range(self.T + 1):
                for m in solution_D[d][t]['pickup']:
                    pickup_d_count[m] += 1
                    if pickup_d_count[m] > 1:
                        return False
                for m in solution_D[d][t]['delivery']:
                    delivery_d_count[m] += 1
                    if delivery_d_count[m] > 1:
                        return False

        # 4. Check displacement constraints
        for k in range(self.K):
            last_t = None
            for t in range(self.T + 1):
                if (t == self.join_time_K[k] or solution_K[k][t]['pickup1'] or solution_K[k][t]['delivery1'] or solution_K[k][t]['pickup2'] or solution_K[k][t]['delivery2']):
                    if last_t is not None:
                        if self.dist[solution_K[k][last_t]['location']][solution_K[k][t]['location']] > t - last_t:
                            return False
                    last_t = t
                    
        for d in range(self.D):
            last_t = None
            for t in range(self.T + 1):
                if (t == self.join_time_D[d] or solution_D[d][t]['pickup'] or solution_D[d][t]['delivery']):
                    if last_t is not None:
                        if self.dist[solution_D[d][last_t]['location']][solution_D[d][t]['location']] > (t - last_t) * self.drone_speed_ratio:
                            return False
                    last_t = t

        # 5. Check location consistency constraints 
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['pickup1']:
                    # 跳过预装载的请求
                    if not self.pre_load_K_stage1[k][m] and solution_K[k][t]['location'] != self.from_req[m]:
                        return False
                for m in solution_K[k][t]['delivery1']:
                    if solution_K[k][t]['location'] != self.station1_req[m]:
                        return False
                for m in solution_K[k][t]['pickup2']:
                    # 跳过预装载的请求
                    if not self.pre_load_K_stage3[k][m] and solution_K[k][t]['location'] != self.station2_req[m]:
                        return False
                for m in solution_K[k][t]['delivery2']:
                    if solution_K[k][t]['location'] != self.to_req[m]:
                        return False

        for d in range(self.D):
            for t in range(self.T + 1):
                for m in solution_D[d][t]['pickup']:
                    # 跳过预装载的请求
                    if not self.pre_load_D[d][m] and solution_D[d][t]['location'] != self.station1_req[m]:
                        return False
                for m in solution_D[d][t]['delivery']:
                    if solution_D[d][t]['location'] != self.station2_req[m]:
                        return False
                    
        # 6. Check that the same courier cannot handle two stages of the same request
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['pickup2']:
                    if self.courier_stage1_temp[m] == k:
                        return False

        # 7. Check request appearance time constraints
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['pickup1']:
                    if t < self.appear[m]:
                        return False

        # 8. Check vehicle start working time constraints
        for k in range(self.K):
            for t in range(self.join_time_K[k]):
                if solution_K[k][t]['delivery1'] or solution_K[k][t]['delivery2']:
                    return False
                elif solution_K[k][t]['pickup1'] or solution_K[k][t]['pickup2']:
                    # Check if they are pre-loaded requests, return False if any are not
                    if t == 0:
                        for m in solution_K[k][t]['pickup1']:
                            if not self.pre_load_K_stage1[k][m]:
                                return False
                        for m in solution_K[k][t]['pickup2']:
                            if not self.pre_load_K_stage3[k][m]:
                                return False
                    else:
                        return False
                    
        for d in range(self.D):
            for t in range(self.join_time_D[d]):
                if solution_D[d][t]['delivery']:
                    return False
                elif solution_D[d][t]['pickup']:
                    # Check if they are pre-loaded requests, return False if any are not
                    if t == 0:
                        for m in solution_D[d][t]['pickup']:
                            if not self.pre_load_D[d][m]:
                                return False
                    else:
                        return False
                
        # 9. Check that orders must be completed in the sequence: Stage1 → Drone → Stage2
        req_times = {m: [-1]*6 for m in range(self.M)}
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in solution_K[k][t]['pickup1']:
                    req_times[m][0] = t
                for m in solution_K[k][t]['delivery1']:
                    req_times[m][1] = t
                for m in solution_K[k][t]['pickup2']:
                    req_times[m][4] = t
                for m in solution_K[k][t]['delivery2']:
                    req_times[m][5] = t
        
        for d in range(self.D):
            for t in range(self.T + 1):
                for m in solution_D[d][t]['pickup']:
                    req_times[m][2] = t
                for m in solution_D[d][t]['delivery']:
                    req_times[m][3] = t
        
        # Check stage sequence constraints
        for m, times in req_times.items():
            t_pu1, t_de1, t_dr_pu, t_dr_de, t_pu2, t_de2 = times
            # Check constraints of Stage 2 (Drone) relative to Stage 1 (Stage1)
            # If Stage1 has pickup1 but no delivery1, Drone cannot have pickup or delivery
            if t_pu1 != -1 and t_de1 == -1:
                if t_dr_pu != -1 or t_dr_de != -1 or t_pu2 != -1 or t_de2 != -1:
                    return False 
            # If Stage1 has delivery1 record, Drone's pickup must be after delivery1
            elif t_de1 != -1:
                if t_dr_pu != -1 and t_dr_pu < t_de1:
                    return False 
            # Check constraints of Stage 3 (Stage2) relative to Stage 2 (Drone)
            # If Drone has pickup but no delivery, Stage2 cannot have pickup2 or delivery2
            if t_dr_pu != -1 and t_dr_de == -1:
                if t_pu2 != -1 or t_de2 != -1:
                    return False 
            # If Drone has delivery record, Stage2's pickup2 must be after delivery
            elif t_dr_de != -1:
                if t_pu2 != -1 and t_pu2 < t_dr_de:
                    return False 
        return True
    
    def find_old_id(self, new_solution_K, new_solution_D, m):
        """Find the old vehicle IDs of request m in each stage"""
        old_k1, old_d, old_k2 = -1, -1, -1
        for k in range(self.K):
            for t in range(self.T + 1):
                if m in new_solution_K[k][t]['pickup1']:
                    old_k1 = k
                if m in new_solution_K[k][t]['delivery1']:
                    old_k1 = k
                if m in new_solution_K[k][t]['pickup2']:
                    old_k2 = k
                if m in new_solution_K[k][t]['delivery2']:
                    old_k2 = k
        for d in range(self.D):
            for t in range(self.T + 1):
                if m in new_solution_D[d][t]['pickup']:
                    old_d = d
                if m in new_solution_D[d][t]['delivery']:
                    old_d = d
        return old_k1, old_d, old_k2
    
    def _remove_request_from_courier(self, solution_K, m, pickup_key, delivery_key):
        """General method: Remove a request from a courier's route"""
        old_k = -1
        for k in range(self.K):
            for t in range(self.T + 1):
                if m in solution_K[k][t][pickup_key]:
                    old_k = k
                    solution_K[k][t][pickup_key].remove(m)
                if m in solution_K[k][t][delivery_key]:
                    old_k = k
                    solution_K[k][t][delivery_key].remove(m)
        return old_k
    
    def _remove_request_from_drone(self, solution_D, m):
        """General method: Remove a request from a drone's route"""
        old_d = -1
        for d in range(self.D):
            for t in range(self.T + 1):
                if m in solution_D[d][t]['pickup']:
                    old_d = d
                    solution_D[d][t]['pickup'].remove(m)
                if m in solution_D[d][t]['delivery']:
                    old_d = d
                    solution_D[d][t]['delivery'].remove(m)
        return old_d
    
    def remove_stage1(self, new_solution_K, new_solution_D, m):
        """Remove Stage 1 operations for request m"""
        return self._remove_request_from_courier(new_solution_K, m, 'pickup1', 'delivery1')
    
    def remove_stage2(self, new_solution_K, new_solution_D, m):
        """Remove Stage 2 operations for request m"""
        return self._remove_request_from_drone(new_solution_D, m)
        
    def remove_stage3(self, new_solution_K, new_solution_D, m):
        """Remove Stage 3 operations for request m"""
        return self._remove_request_from_courier(new_solution_K, m, 'pickup2', 'delivery2')

    def reassign_stage1(self, new_solution_K, new_solution_D, m, end_time):
        """Reassign Stage 1 for request m"""
        old_k1 = self.remove_stage1(new_solution_K, new_solution_D, m)
        if old_k1 != -1 and self.pre_load_K_stage1[old_k1][m]:
            k = old_k1
            new_solution_K[k][0]['pickup1'].append(m)
            if self.dist[self.start_K[k]][self.station1_req[m]] + self.join_time_K[k] <= end_time:
                t_delivery = self.rng.integers(self.dist[self.start_K[k]][self.station1_req[m]] + self.join_time_K[k], end_time + 1)
                new_solution_K[k][t_delivery]['delivery1'].append(m)
        elif self.req_cur_stage[m] < 1:
            k = self.rng.integers(0, self.K)
            if max(self.join_time_K[k], self.appear[m]) >= end_time + 1:
                end_time = self.T
            t_pickup = self.rng.integers(max(self.join_time_K[k], self.appear[m]), end_time + 1)
            new_solution_K[k][t_pickup]['pickup1'].append(m)
            if t_pickup + self.dist[self.from_req[m]][self.station1_req[m]] <= end_time:
                t_delivery = self.rng.integers(t_pickup + self.dist[self.from_req[m]][self.station1_req[m]], end_time + 1)
                new_solution_K[k][t_delivery]['delivery1'].append(m)
            
    def reassign_stage2(self, new_solution_K, new_solution_D, m, end_time):
        """Reassign Stage 2 for request m"""
        old_d = self.remove_stage2(new_solution_K, new_solution_D, m)
        if old_d != -1 and self.pre_load_D[old_d][m]:
            d = old_d
            new_solution_D[d][0]['pickup'].append(m)
            if self.dist[self.start_D[d]][self.station2_req[m]] + self.join_time_D[d] <= end_time:
                t_delivery = min(int(np.ceil(self.dist[self.start_D[d]][self.station2_req[m]] / self.drone_speed_ratio) + self.join_time_D[d]), end_time)
                new_solution_D[d][t_delivery]['delivery'].append(m)
        elif self.req_cur_stage[m] < 2:
            t_delivery1 = self.find_delivery1_time(new_solution_K, m)
            if t_delivery1 is None:
                self.reassign_stage1(new_solution_K, new_solution_D, m, end_time/2+1)
                t_delivery1 = self.find_delivery1_time(new_solution_K, m)
                if t_delivery1 is None:
                    return
            d = self.rng.integers(0, self.D)
            if max(t_delivery1, self.join_time_D[d]) >= end_time + 1:
                end_time = self.T
            t_pickup_d = self.rng.integers(max(t_delivery1, self.join_time_D[d]), end_time + 1)
            new_solution_D[d][t_pickup_d]['pickup'].append(m)
            if t_pickup_d + np.ceil(self.dist[self.station1_req[m]][self.station2_req[m]] / self.drone_speed_ratio) <= end_time:
                t_delivery = min(int(t_pickup_d + np.ceil(self.dist[self.station1_req[m]][self.station2_req[m]] / self.drone_speed_ratio)), end_time)
                new_solution_D[d][t_delivery]['delivery'].append(m)
        
    def reassign_stage3(self, new_solution_K, new_solution_D, m, end_time):
        """Reassign Stage 3 for request m"""
        old_k2 = self.remove_stage3(new_solution_K, new_solution_D, m)
        if old_k2 != -1 and self.pre_load_K_stage3[old_k2][m]:
            k = old_k2
            new_solution_K[k][0]['pickup2'].append(m)
            if self.dist[self.start_K[k]][self.to_req[m]] + self.join_time_K[k] <= end_time:
                t_delivery = self.rng.integers(self.dist[self.start_K[k]][self.to_req[m]] + self.join_time_K[k], end_time + 1)
                new_solution_K[k][t_delivery]['delivery2'].append(m)
        elif self.req_cur_stage[m] < 3:
            t_delivery_d = self.find_delivery_d_time(new_solution_D, m)
            if t_delivery_d is None:
                self.reassign_stage2(new_solution_K, new_solution_D, m, end_time)
                t_delivery_d = self.find_delivery_d_time(new_solution_D, m)
                if t_delivery_d is None:
                    return
            k = self.rng.integers(0, self.K)
            if max(self.join_time_K[k], t_delivery_d) >= end_time + 1:
                end_time = self.T
            t_pickup = self.rng.integers(max(self.join_time_K[k], t_delivery_d), end_time + 1)
            new_solution_K[k][t_pickup]['pickup2'].append(m)
            if t_pickup + self.dist[self.station2_req[m]][self.to_req[m]] <= end_time:
                t_delivery = self.rng.integers(t_pickup + self.dist[self.station2_req[m]][self.to_req[m]], end_time + 1)
                new_solution_K[k][t_delivery]['delivery2'].append(m)
    
    def find_pickup_d_time(self, solution_D, m):
        """Find the start time of Stage 2 for request m"""
        for d in range(self.D):
            for t in range(self.T + 1):
                if m in solution_D[d][t]['pickup']:
                    return t
        return None

    def find_pickup2_time(self, solution_K, m):
        """Find the start time of Stage 3 for request m"""
        for k in range(self.K):
            for t in range(self.T + 1):
                if m in solution_K[k][t]['pickup2']:
                    return t
        return None
    
    def find_delivery1_time(self, solution_K, m):
        """Find the completion time of Stage 1 for request m"""
        if self.wait_stage2[m]:
            return 0
        for k in range(self.K):
            for t in range(self.T + 1):
                if m in solution_K[k][t]['delivery1']:
                    return t
        return None

    def find_delivery_d_time(self, solution_D, m):
        """Find the completion time of Stage 2 for request m"""
        if self.wait_stage3[m]:
            return 0
        for d in range(self.D):
            for t in range(self.T + 1):
                if m in solution_D[d][t]['delivery']:
                    return t
        return None
    
    def find_delivery2_time(self, solution_K, m):
        """Find the completion time of Stage 3 for request m"""
        for k in range(self.K):
            for t in range(self.T + 1):
                if m in solution_K[k][t]['delivery2']:
                    return t
        return None

    def initial_solution(self):
        """Generate initial solution"""
        # Initial solution: randomly assign requests to vehicles
        solution_K = []
        solution_D = []
        for k in range(self.K):
            path_K = []
            for t in range(0, self.T + 1):
                path_K.append({'location': self.start_K[k], 'pickup1': [], 'delivery1': [], 'pickup2': [], 'delivery2': [], 'load': 0})
            solution_K.append(path_K)

        for d in range(self.D):
            path_D = []
            for t in range(0, self.T + 1):
                path_D.append({'location': self.start_D[d], 'pickup': [], 'delivery': [], 'load': 0})
            solution_D.append(path_D)
        
        # Process pre-loaded requests, record as picked up by vehicle k at time 0
        for m in range(self.M):
            for k in range(self.K):
                if self.pre_load_K_stage1[k][m]:
                    self.req_cur_stage[m] = 1
                    solution_K[k][0]['pickup1'].append(m)
                elif self.pre_load_K_stage3[k][m]:
                    self.req_cur_stage[m] = 3
                    solution_K[k][0]['pickup2'].append(m)          
            for d in range(self.D):
                if self.pre_load_D[d][m]:
                    self.req_cur_stage[m] = 2
                    solution_D[d][0]['pickup'].append(m)
            if self.wait_stage2[m]:
                self.req_cur_stage[m] = 1
            elif self.wait_stage3[m]:
                self.req_cur_stage[m] = 2
        # Update positions and loads of all vehicles
        self._update_all_vehicles(solution_K, solution_D)
        
        if self.check_constraints(solution_K, solution_D):
            return solution_K, solution_D
        else:
            return [], []
    
    def _convert_solution_to_output(self, best_solution_K, best_solution_D):
        """
        Convert solution to output format
        Returns: (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)
        """
        _pickup1 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup2 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup_d = np.zeros((self.D, self.M, self.T + 1), dtype=bool)
        _location_k = np.full((self.K, self.T + 1), -1, dtype=int)
        _location_k[np.arange(self.K), self.join_time_K] = self.start_K
        _location_d = np.full((self.D, self.T + 1), -1, dtype=int)
        _location_d[np.arange(self.D), self.join_time_D] = self.start_D
        
        for k in range(self.K):
            for t in range(self.T + 1):
                for m in best_solution_K[k][t]['pickup1']:
                    if not self.pre_load_K_stage1[k][m] and not self.wait_stage2[m]:
                        _pickup1[k, m, t] = True
                        assert _location_k[k, t] == -1 or _location_k[k, t] == self.from_req[m]
                        _location_k[k, t] = self.from_req[m]
                for m in best_solution_K[k][t]['delivery1']:
                    assert _location_k[k, t] == -1 or _location_k[k, t] == self.station1_req[m]
                    _location_k[k, t] = self.station1_req[m]
                for m in best_solution_K[k][t]['pickup2']:
                    if not self.pre_load_K_stage3[k][m]:
                        _pickup2[k, m, t] = True
                        assert _location_k[k, t] == -1 or _location_k[k, t] == self.station2_req[m]
                        _location_k[k, t] = self.station2_req[m]
                for m in best_solution_K[k][t]['delivery2']:
                    assert _location_k[k, t] == -1 or _location_k[k, t] == self.to_req[m]
                    _location_k[k, t] = self.to_req[m]
        
        for d in range(self.D):
            for t in range(self.T + 1):
                for m in best_solution_D[d][t]['pickup']:
                    if not self.pre_load_D[d][m] and not self.wait_stage3[m]:
                        _pickup_d[d, m, t] = True
                        assert _location_d[d, t] == -1 or _location_d[d, t] == self.station1_req[m]
                        _location_d[d, t] = self.station1_req[m]
                for m in best_solution_D[d][t]['delivery']:
                    assert _location_d[d, t] == -1 or _location_d[d, t] == self.station2_req[m]
                    _location_d[d, t] = self.station2_req[m]
        
        return _pickup1, _pickup2, _pickup_d, _location_k, _location_d
    
    def _print_solution(self, best_solution_K, best_solution_D, best_obj, is_genetic=False):
        """
        Print solution results
        :param best_solution_K: Courier routes of the best solution
        :param best_solution_D: Drone routes of the best solution
        :param best_obj: Best objective function value
        :param is_genetic: Whether it is a genetic algorithm (modified to directly use objective function value, no need to take logarithm)
        """
        # Now both genetic algorithm and simulated annealing directly use objective function value, unified printing
        print("Best Objective Value:", best_obj)
        
        print("=======================Request Result=======================")
        for m in range(self.M):
            print(f"Request {m} From: {self.from_req[m]}, To: {self.to_req[m]}, Station1: {self.station1_req[m]}, Station2: {self.station2_req[m]}, Appear: {self.appear[m]}, Value: {self.value[m]}, Penalty: {self.penalty[m]}")
            for k in range(self.K):
                for t in range(self.T + 1):
                    if m in best_solution_K[k][t]['pickup1']:
                        print(f"Courier {k} Pickup1 Request {m} at t = {t}")
                    if m in best_solution_K[k][t]['delivery1']:
                        print(f"Courier {k} Delivery1 Request {m} at t = {t}")
                    if m in best_solution_K[k][t]['pickup2']:
                        print(f"Courier {k} Pickup2 Request {m} at t = {t}")
                    if m in best_solution_K[k][t]['delivery2']:
                        print(f"Courier {k} Delivery2 Request {m} at t = {t}")
            for d in range(self.D):
                for t in range(self.T + 1):
                    if m in best_solution_D[d][t]['pickup']:
                        print(f"Drone {d} Pickup Request {m} at t = {t}")
                    if m in best_solution_D[d][t]['delivery']:
                        print(f"Drone {d} Delivery Request {m} at t = {t}")
        
        print("=======================Vehicle Result=======================")
        for k in range(self.K):
            print(f"Courier {k} Route:")
            for t in range(self.T + 1):
                print(
                    f"t = {t}, location = {best_solution_K[k][t]['location']}, load = {best_solution_K[k][t]['load']}, pickup1 = {best_solution_K[k][t]['pickup1']}, delivery1 = {best_solution_K[k][t]['delivery1']}, pickup2 = {best_solution_K[k][t]['pickup2']}, delivery2 = {best_solution_K[k][t]['delivery2']}")
        for d in range(self.D):
            print(f"Drone {d} Route:")
            for t in range(self.T + 1):
                print(
                    f"t = {t}, location = {best_solution_D[d][t]['location']}, pickup = {best_solution_D[d][t]['pickup']}, delivery = {best_solution_D[d][t]['delivery']}")


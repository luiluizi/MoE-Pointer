import numpy as np

class Request:
    def __init__(self, req_id, from_node, to_node, station1_node, station2_node, appear_time, serving_courier_stage1):
        self.req_id = req_id
        self.from_node = from_node
        self.to_node = to_node
        self.station1_node = station1_node
        self.station2_node = station2_node
        self.appear_time = appear_time
        self.picked_up_stage1 = False
        self.picked_up_stage2 = False
        self.picked_up_stage3 = False
        self.delivered_stage1 = False
        self.delivered_stage2 = False
        self.delivered_stage3 = False
        self.pickup_time_stage1 = -1
        self.pickup_time_stage2 = -1
        self.pickup_time_stage3 = -1
        self.delivery_time_stage1 = -1
        self.delivery_time_stage2 = -1
        self.delivery_time_stage3 = -1
        self.serving_courier_stage1 = serving_courier_stage1
        self.serving_courier_stage3 = -1
        self.serving_drone = -1
        self.stage = 0
        
class Courier:
    def __init__(self, courier_id, start_node, capacity, join_time, dist_matrix, cost, time_limit):
        self.courier_id = courier_id
        self.current_node = start_node
        self.capacity = capacity
        self.load = [0]
        self.route = [start_node]
        self.times = [join_time]  # Time at each node
        self.dist_matrix = dist_matrix
        self.cost_matrix = cost
        self.available_time = join_time
        self.join_time = join_time
        self.served_requests_stage1 = []
        self.served_requests_stage3 = []
        self.to_pickup_requests_stage1 = []
        self.to_delivery_requests_stage1 = []
        self.to_pickup_requests_stage3 = []
        self.to_delivery_requests_stage3 = []
        self.time_limit = time_limit

    def can_pickup(self, request, stage_id):
        # Check if the vehicle can reach the request node and perform pickup
        if stage_id == 1:
            travel_time = self.dist_matrix[self.current_node][request.from_node]
            cost = self.cost_matrix[self.current_node][request.from_node]
        elif stage_id == 3:
            travel_time = self.dist_matrix[self.current_node][request.station2_node]
            cost = self.cost_matrix[self.current_node][request.station2_node]
            if request.serving_courier_stage1 == self.courier_id:
                return False, travel_time, cost
        pickup_time = self.available_time + travel_time
        
        request_cur_stage_time = request.appear_time if stage_id == 1 else request.delivery_time_stage2
        if request_cur_stage_time <= pickup_time <= self.time_limit and self.load[-1] < self.capacity:
            return True, travel_time, cost
        return False, travel_time, cost

    def pickup_request(self, request, stage_id, consider_others=True):
        # Travel to the request node for pickup
        # Update vehicle status
        from_node = request.from_node if stage_id == 1 else request.station2_node
        
        self.available_time += self.dist_matrix[self.current_node][from_node]
        if self.current_node != from_node:
            self.route.append(from_node)
            self.current_node = from_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] + 1)
        else:
            self.load[-1] += 1
        
        if stage_id == 1:
            self.to_pickup_requests_stage1.remove(request)
            # Update request status
            request.picked_up_stage1 = True
            request.pickup_time_stage1 = self.times[-1]
            request.serving_courier_stage1 = self.courier_id
            # Add to pending delivery list
            self.to_delivery_requests_stage1.append(request)
        elif stage_id == 3:
            self.to_pickup_requests_stage3.remove(request)
            # Update request status
            request.picked_up_stage3 = True
            request.pickup_time_stage3 = self.times[-1]
            request.serving_courier_stage3 = self.courier_id
            # Add to pending delivery list
            self.to_delivery_requests_stage3.append(request)
        
        if not consider_others:
            return
        # Check for other requests at the current node
        for req in self.to_pickup_requests_stage1.copy():
            if req.from_node == self.current_node and self.can_pickup(req, 1)[0]:
                self.pickup_request(req, 1, consider_others=False)
        for req in self.to_pickup_requests_stage3.copy():
            if req.station2_node == self.current_node and self.can_pickup(req, 3)[0]:
                self.pickup_request(req, 3, consider_others=False)
        for req in self.to_delivery_requests_stage1.copy():
            if req.station1_node == self.current_node:
                self.delivery_request(req, 1, consider_others=False)
        for req in self.to_delivery_requests_stage3.copy():
            if req.to_node == self.current_node:
                self.delivery_request(req, 3, consider_others=False)

    def can_delivery(self, request, stage_id):
        # Check if the request can be delivered at present
        if stage_id == 1:
            travel_time = self.dist_matrix[self.current_node][request.station1_node]
            cost = self.cost_matrix[self.current_node][request.station1_node]
        elif stage_id == 3:
            travel_time = self.dist_matrix[self.current_node][request.to_node]
            cost = self.cost_matrix[self.current_node][request.to_node]
        delivery_time = self.available_time + travel_time
        if delivery_time <= self.time_limit:
            return True, travel_time, cost
        return False, travel_time, cost

    def delivery_request(self, request, stage_id, consider_others=True):
        # Travel to the request node for delivery
        # Update vehicle status
        to_node = request.station1_node if stage_id == 1 else request.to_node
        self.available_time += self.dist_matrix[self.current_node][to_node]
        if self.current_node != to_node:
            self.route.append(to_node)
            self.current_node = to_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] - 1)
        else:
            self.load[-1] -= 1
        
        if stage_id == 1:
            self.to_delivery_requests_stage1.remove(request)
            # Update request status
            request.delivered_stage1 = True
            request.delivery_time_stage1 = self.times[-1]
            request.serving_courier_stage1 = self.courier_id
            # Add to served requests list
            self.served_requests_stage1.append(request)
        elif stage_id == 3:
            self.to_delivery_requests_stage3.remove(request)
            # Update request status
            request.delivered_stage3 = True
            request.delivery_time_stage3 = self.times[-1]
            request.serving_courier_stage3 = self.courier_id
            # Add to served requests list
            self.served_requests_stage3.append(request)
        if not consider_others:
            return
        # Check for other requests at the current node
        for req in self.to_pickup_requests_stage1.copy():
            if req.from_node == self.current_node and self.can_pickup(req, 1)[0]:
                self.pickup_request(req, 1, consider_others=False)
        for req in self.to_pickup_requests_stage3.copy():
            if req.station2_node == self.current_node and self.can_pickup(req, 3)[0]:
                self.pickup_request(req, 3, consider_others=False)
        for req in self.to_delivery_requests_stage1.copy():
            if req.station1_node == self.current_node:
                self.delivery_request(req, 1, consider_others=False)
        for req in self.to_delivery_requests_stage3.copy():
            if req.to_node == self.current_node:
                self.delivery_request(req, 3, consider_others=False)

class Drone:
    def __init__(self, drone_id, start_node, join_time, dist_matrix, cost, time_limit, drone_speed_ratio):
        self.drone_id = drone_id
        self.current_node = start_node
        self.capacity = 1
        self.load = [0]
        self.route = [start_node]
        self.times = [join_time]  # Time at each node
        self.dist_matrix = dist_matrix
        self.cost_matrix = cost
        self.available_time = join_time
        self.join_time = join_time
        self.served_requests = []
        self.to_pickup_requests = []
        self.to_delivery_requests = []
        self.time_limit = time_limit
        self.drone_speed_ratio = drone_speed_ratio

    def can_pickup(self, request):
        # Check if the drone can reach the request node and perform pickup
        travel_time = np.ceil(self.dist_matrix[self.current_node][request.station1_node] / self.drone_speed_ratio).astype(int)
        pickup_time = self.available_time + travel_time
        cost = self.cost_matrix[self.current_node][request.station1_node]
        if request.delivery_time_stage1 <= pickup_time <= self.time_limit and self.load[-1] < self.capacity:
            return True, travel_time, cost
        return False, travel_time, cost

    def pickup_request(self, request, consider_others=True):
        # Travel to the request node for pickup
    # Update drone status
        self.available_time += np.ceil(self.dist_matrix[self.current_node][request.station1_node] / self.drone_speed_ratio).astype(int)
        if self.current_node != request.station1_node:
            self.route.append(request.station1_node)
            self.current_node = request.station1_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] + 1)
        else:
            self.load[-1] += 1
        self.to_pickup_requests.remove(request)

        # Update request status
        request.picked_up_stage2 = True
        request.pickup_time_stage2 = self.times[-1]
        request.serving_drone = self.drone_id

        # Add to pending delivery list
        self.to_delivery_requests.append(request)

        if not consider_others:
            return

    def can_delivery(self, request):
        # Check if the request can be delivered at present
        travel_time = np.ceil(self.dist_matrix[self.current_node][request.station2_node] / self.drone_speed_ratio).astype(int)
        delivery_time = self.available_time + travel_time
        cost = self.cost_matrix[self.current_node][request.station2_node]

        if delivery_time <= self.time_limit:
            return True, travel_time, cost
        return False, travel_time, cost

    def delivery_request(self, request, consider_others=True):
        # Travel to the request node for delivery
        # Update drone status
        self.available_time += np.ceil(self.dist_matrix[self.current_node][request.station2_node] / self.drone_speed_ratio).astype(int)
        if self.current_node != request.station2_node:
            self.route.append(request.station2_node)
            self.current_node = request.station2_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] - 1)
        else:
            self.load[-1] -= 1
        self.to_delivery_requests.remove(request)
        # Update request status
        request.delivered_stage2 = True
        request.delivery_time_stage2 = self.times[-1]
        request.serving_drone = self.drone_id
        # Add to served requests list
        self.served_requests.append(request)
        if not consider_others:
            return

class NearestHeuristic:
    def __init__(self, N, M, K, D, T, start_K, start_D, capacity, join_time_K, join_time_D, dist, 
                 cost_K, cost_D, from_req, to_req, station1_req, station2_req, appear, value, penalty, 
                 pre_load_K_stage1, pre_load_K_stage3, pre_load_D, wait_stage2, wait_stage3, drone_speed_ratio=4.0, courier_stage1_temp=None, **kwargs):
        """
        :param N: Number of nodes
        :param M: Number of requests
        :param K: Number of couriers
        :param D: Number of drones
        :param T: Number of time frames
        :param start_K: Starting point of each courier
        :param start_D: Starting point of each drone
        :param capacity: Capacity of each vehicle
        :param join_time_K: Start working time of each courier, used for couriers with non-zero time_left
        :param join_time_D: Start working time of each drone, used for drones with non-zero time_left
        :param dist: Distance matrix
        :param cost_K: Cost matrix for couriers
        :param cost_D: Cost matrix for drones
        :param from_req: Starting point of each request
        :param to_req: Destination of each request
        :param station1_req: Station 1 of each request
        :param station2_req: Station 2 of each request
        :param appear: Appearance time of each request
        :param value: Value of each request
        :param penalty: Penalty for each request
        :param pre_load_K_stage1: K x M matrix, indicating whether a request has been pre-loaded on a courier in Stage 1
        :param pre_load_K_stage3: K x M matrix, indicating whether a request has been pre-loaded on a courier in Stage 3
        :param pre_load_D: D x M matrix, indicating whether a request has been pre-loaded on a drone
        :param wait_stage2: M-dimensional array, indicating whether a request is waiting to enter Stage 2
        :param wait_stage3: M-dimensional array, indicating whether a request is waiting to enter Stage 3
        :param courier_stage1_temp: Records the courier that completed Stage 1 delivery for each order, to prevent the same courier from being used for both Stage 1 and Stage 3
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
        self.couriers = []
        self.drones = []
        self.requests = []
        self.courier_stage1_temp = courier_stage1_temp

    def initialize_requests_and_vehicles(self):
        for m in range(self.M):
            courier_stage1_temp = self.courier_stage1_temp[m] if self.courier_stage1_temp[m] != self.K else -1
            self.requests.append(Request(m, self.from_req[m], self.to_req[m], self.station1_req[m], self.station2_req[m], self.appear[m], courier_stage1_temp))
        for k in range(self.K):
            self.couriers.append(Courier(k, self.start_K[k], self.capacity[k], self.join_time_K[k], self.dist, self.cost_K, self.T))
        for d in range(self.D):
            self.drones.append(Drone(d, self.start_D[d], self.join_time_D[d], self.dist, self.cost_D, self.T, self.drone_speed_ratio))
        # Process pre-loaded requests
        for m in range(self.M):
            for k in range(self.K):
                if self.pre_load_K_stage1[k][m]:
                    self.couriers[k].load[-1] += 1
                    self.couriers[k].to_delivery_requests_stage1.append(self.requests[m])
                    self.requests[m].picked_up_stage1 = True
                    self.requests[m].pickup_time_stage1 = self.join_time_K[k]
                    self.requests[m].serving_courier_stage1 = k
                    self.requests[m].stage = 1
                elif self.pre_load_K_stage3[k][m]:
                    self.couriers[k].load[-1] += 1
                    self.couriers[k].to_delivery_requests_stage3.append(self.requests[m])
                    self.requests[m].picked_up_stage1 = True
                    self.requests[m].picked_up_stage2 = True
                    self.requests[m].delivered_stage1 = True
                    self.requests[m].delivered_stage2 = True
                    self.requests[m].picked_up_stage3 = True
                    self.requests[m].pickup_time_stage3 = self.join_time_K[k]
                    self.requests[m].serving_courier_stage3 = k
                    self.requests[m].stage = 3
            for d in range(self.D):
                if self.pre_load_D[d][m]:
                    self.drones[d].load[-1] += 1
                    self.drones[d].to_delivery_requests.append(self.requests[m])
                    self.requests[m].picked_up_stage1 = True
                    self.requests[m].delivered_stage1 = True
                    self.requests[m].picked_up_stage2 = True
                    self.requests[m].pickup_time_stage2 = self.join_time_D[d]
                    self.requests[m].serving_drone = d
                    self.requests[m].stage = 2
            if self.wait_stage2[m]:
                self.requests[m].picked_up_stage1 = True
                self.requests[m].delivered_stage1 = True
                self.requests[m].stage = 1
            elif self.wait_stage3[m]:
                self.requests[m].picked_up_stage1 = True
                self.requests[m].picked_up_stage2 = True
                self.requests[m].delivered_stage1 = True
                self.requests[m].delivered_stage2 = True
                self.requests[m].stage = 2
        
    def objective_function(self):
        total_value = 0
        total_cost = 0
        total_penalty = 0
        # Calculate total value
        for req in self.requests:
            if req.delivered_stage1:
                total_value += self.value[req.req_id]
            if req.delivered_stage2:
                total_value += self.value[req.req_id]
            if req.delivered_stage3:
                total_value += self.value[req.req_id]
        # Calculate total cost
        for cou in self.couriers:
            for i in range(len(cou.route)-1):
                total_cost += self.cost_K[cou.route[i]][cou.route[i+1]]
        for dro in self.drones:
            for i in range(len(dro.route)-1):
                total_cost += self.cost_D[dro.route[i]][dro.route[i+1]]
        # Calculate total penalty
        for req in self.requests:
            if not req.delivered_stage3:
                total_penalty += self.penalty[req.req_id]
        return total_value - total_cost - total_penalty
    
    def request_assign_courier(self, requests, t, stage_id):
        request_removing = []
        for req in requests:
            appear_time = req.appear_time if stage_id == 1 else req.delivery_time_stage2
            
            if stage_id == 3:
                assert(req.delivered_stage2)
            
            if appear_time <= t:
                # Try to assign to the courier with the minimum cost that can accept the request
                min_cost = np.inf
                selected_courier = None
                selected_travel_time = 0
                for cou in self.couriers:
                    if t < cou.join_time:
                        continue
                    can_pick, travel_time, cost = cou.can_pickup(req, stage_id)
                    if can_pick:
                        if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                            min_cost = cost
                            selected_courier = cou
                            selected_travel_time = travel_time
                if selected_courier:
                    if stage_id == 1:
                        selected_courier.to_pickup_requests_stage1.append(req)
                    elif stage_id == 3:
                        selected_courier.to_pickup_requests_stage3.append(req)
                    request_removing.append(req)
                    
        for req in request_removing:
            requests.remove(req)
        return requests
    
    def request_assign_drone(self, requests, t):
        request_removing = []
        for req in requests:
            if req.delivery_time_stage1 <= t:
                # Try to assign to the drone with the minimum cost that can accept the request
                assert(req.delivered_stage1)
                min_cost = np.inf
                selected_drone = None
                selected_travel_time = 0
                for dro in self.drones:
                    if t < dro.join_time:
                        continue
                    can_pick, travel_time, cost = dro.can_pickup(req)
                    if can_pick:
                        if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                            min_cost = cost
                            selected_drone = dro
                            selected_travel_time = travel_time
                if selected_drone:
                    selected_drone.to_pickup_requests.append(req)
                    request_removing.append(req)
        for req in request_removing:
            requests.remove(req)
        return requests

    def courier_action(self, t, stage_id, next_stage_requests):
        for cou in self.couriers:
            if t < cou.available_time:
                continue
            # Check for requests requiring delivery and select the one with the minimum cost
            min_cost = np.inf
            selected_request = None
            selected_travel_time = 0
            delivery_requests = cou.to_delivery_requests_stage1 if stage_id == 1 else cou.to_delivery_requests_stage3
            pickup_requests = cou.to_pickup_requests_stage1 if stage_id == 1 else cou.to_pickup_requests_stage3
            
            for req in delivery_requests:
                can_delivery, travel_time, cost = cou.can_delivery(req, stage_id)
                if can_delivery:
                    if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                        min_cost = cost
                        selected_request = req
                        selected_travel_time = travel_time
            if selected_request:
                cou.delivery_request(selected_request, stage_id)
                next_stage_requests.append(selected_request)
            # Check for requests requiring pickup and select the one with the minimum cost
            min_cost = np.inf
            selected_request = None
            selected_travel_time = 0
            for req in pickup_requests:
                can_pickup, travel_time, cost = cou.can_pickup(req, stage_id)
                if can_pickup:
                    if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                        min_cost = cost
                        selected_request = req
                        selected_travel_time = travel_time
            if selected_request:
                cou.pickup_request(selected_request, stage_id)
        return next_stage_requests
    
    def drone_action(self, t, next_stage_requests):
        for dro in self.drones:
            if t < dro.available_time:
                continue
            # Check for requests requiring delivery and select the one with the minimum cost
            min_cost = np.inf
            selected_request = None
            selected_travel_time = 0
            for req in dro.to_delivery_requests:
                can_delivery, travel_time, cost = dro.can_delivery(req)
                if can_delivery:
                    if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                        min_cost = cost
                        selected_request = req
                        selected_travel_time = travel_time
            if selected_request:
                dro.delivery_request(selected_request)
                next_stage_requests.append(selected_request)
            # Check for requests requiring pickup and select the one with the minimum cost
            min_cost = np.inf
            selected_request = None
            selected_travel_time = 0
            for req in dro.to_pickup_requests:
                can_pickup, travel_time, cost = dro.can_pickup(req)
                if can_pickup:
                    if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                        min_cost = cost
                        selected_request = req
                        selected_travel_time = travel_time
            if selected_request:
                dro.pickup_request(selected_request)
        return next_stage_requests 
        
    def solve(self, show=True):
        self.initialize_requests_and_vehicles()
        request_stage1 = []
        request_stage2 = []
        request_stage3 = []
        # Process pre-loaded requests
        for req in self.requests:
            if not req.picked_up_stage1:
                request_stage1.append(req)
            elif (not req.picked_up_stage2) and req.delivered_stage1:
                request_stage2.append(req)
            elif (not req.picked_up_stage3) and req.delivered_stage2:
                request_stage3.append(req)    
        request_stage1 = sorted(request_stage1, key=lambda x: x.appear_time)
        
        # Assign pickups and deliveries in chronological order
        for t in range(self.T + 1):
            # Stage 1 request
            request_stage1 = self.request_assign_courier(request_stage1, t, 1)
            # Stage 2 request
            request_stage2 = self.request_assign_drone(request_stage2, t)
            # Stage 3 request
            request_stage3 = self.request_assign_courier(request_stage3, t, 3)
            # courier action
            request_stage2 = self.courier_action(t, 1, request_stage2)
            # drone action
            request_stage3 = self.drone_action(t, request_stage3)
            
            self.courier_action(t, 3, [])
            
        # calculate objective function
        objective_value = self.objective_function()

        _pickup1 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup2 = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _pickup_d = np.zeros((self.D, self.M, self.T + 1), dtype=bool)
        _location_k = np.full((self.K, self.T + 1), -1, dtype=int)
        _location_d = np.full((self.D, self.T + 1), -1, dtype=int)

        for req in self.requests:
            if req.picked_up_stage1:
                if req.stage < 1 and req.serving_courier_stage1 != -1 and req.pickup_time_stage1 != -1:
                    _pickup1[req.serving_courier_stage1, req.req_id, req.pickup_time_stage1] = True
            if req.picked_up_stage2:
                if req.stage < 2 and req.serving_drone != -1 and req.pickup_time_stage2 != -1:
                    _pickup_d[req.serving_drone, req.req_id, req.pickup_time_stage2] = True
            if req.picked_up_stage3:
                if req.stage < 3 and req.serving_courier_stage3 != -1 and req.pickup_time_stage3 != -1:
                    _pickup2[req.serving_courier_stage3, req.req_id, req.pickup_time_stage3] = True
        
        for cou in self.couriers:
            for t, loc in zip(cou.times, cou.route):
                _location_k[cou.courier_id, t] = loc
        
        for dro in self.drones:
            for t, loc in zip(dro.times, dro.route):
                _location_d[dro.drone_id, t] = loc
        
        if not show:
            return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)

        print("Objective Value:", objective_value)
        print("=======================Request Result=======================")
        for req in self.requests:
            if req.req_id == 25:
                print(f"Request {req.req_id} From: {req.from_node} to: {req.to_node} Station1: {req.station1_node} Station2: {req.station2_node}, Appear: {req.appear_time}, Value: {self.value[req.req_id]}, Penalty: {self.penalty[req.req_id]}, Stage: {req.stage}")
                if req.picked_up_stage1:
                    print(f"Courier {req.serving_courier_stage1} picked up 1 at t={req.pickup_time_stage1}")
                if req.delivered_stage1:
                    print(f"Courier {req.serving_courier_stage1} delivered 1 at t={req.delivery_time_stage1}")
                if req.picked_up_stage2:
                    print(f"Drone {req.serving_drone} picked up at t={req.pickup_time_stage2}")
                if req.delivered_stage2:
                    print(f"Drone {req.serving_drone} delivered at t={req.delivery_time_stage2}")
                if req.picked_up_stage3:
                    print(f"Courier {req.serving_courier_stage3} picked up 2 at t={req.pickup_time_stage3}")
                if req.delivered_stage3:
                    print(f"Courier {req.serving_courier_stage3} delivered 2 at t={req.delivery_time_stage3}")
        print("=======================Vehicle Result=======================")
        for cou in self.couriers:
            print(f"Courier {cou.courier_id} Route:")
            for i, time in enumerate(cou.times):
                pick_up_1 = [req.req_id for req in self.requests if req.serving_courier_stage1 == cou.courier_id and req.pickup_time_stage1 == time]
                delivery_1 = [req.req_id for req in self.requests if req.serving_courier_stage1 == cou.courier_id and req.delivery_time_stage1 == time]
                print(f"t = {time}, location = {cou.route[i]}, load = {cou.load[i]}, pick up 1 = {pick_up_1}, delivery 1 = {delivery_1}")
                pick_up_2 = [req.req_id for req in self.requests if req.serving_courier_stage3 == cou.courier_id and req.pickup_time_stage3 == time]
                delivery_2 = [req.req_id for req in self.requests if req.serving_courier_stage3 == cou.courier_id and req.delivery_time_stage3 == time]
                print(f"t = {time}, location = {cou.route[i]}, load = {cou.load[i]}, pick up 2 = {pick_up_2}, delivery 2 = {delivery_2}")
                
        for dro in self.drones:
            print(f"Drone {dro.drone_id} Route:")
            for i, time in enumerate(dro.times):
                pick_up = [req.req_id for req in self.requests if req.serving_drone == dro.drone_id and req.pickup_time_stage2 == time]
                delivery = [req.req_id for req in self.requests if req.serving_drone == dro.drone_id and req.delivery_time_stage2 == time]
                print(f"t = {time}, location = {dro.route[i]}, load = {dro.load[i]}, pick up = {pick_up}, delivery = {delivery}")
        
        return (_pickup1, _pickup2, _pickup_d, _location_k, _location_d)
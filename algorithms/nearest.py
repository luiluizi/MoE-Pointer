import numpy as np

class Request:
    def __init__(self, req_id, from_node, to_node, appear_time):
        self.req_id = req_id
        self.from_node = from_node
        self.to_node = to_node
        self.appear_time = appear_time
        self.picked_up = False
        self.delivered = False
        self.pickup_time = -1
        self.delivery_time = -1
        self.serving_vehicle = None

class Vehicle:
    def __init__(self, vehicle_id, start_node, capacity, join_time, dist_matrix, cost, time_limit):
        self.vehicle_id = vehicle_id
        self.current_node = start_node
        self.capacity = capacity
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

    def can_pickup(self, request):
        # 能否到达请求节点并 pickup
        travel_time = self.dist_matrix[self.current_node][request.from_node]
        pickup_time = self.available_time + travel_time
        cost = self.cost_matrix[self.current_node][request.from_node]
        if request.appear_time <= pickup_time <= self.time_limit and self.load[-1] < self.capacity:
            return True, travel_time, cost
        return False, travel_time, cost

    def pickup_request(self, request, consider_others=True):
        # 前往请求节点 pickup
        # 更新车辆状态
        self.available_time += self.dist_matrix[self.current_node][request.from_node]
        if self.current_node != request.from_node:
            self.route.append(request.from_node)
            self.current_node = request.from_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] + 1)
        else:
            self.load[-1] += 1
        self.to_pickup_requests.remove(request)

        # 更新请求状态
        request.picked_up = True
        request.pickup_time = self.times[-1]
        request.serving_vehicle = self.vehicle_id

        # 放入待delivery列表
        self.to_delivery_requests.append(request)

        if not consider_others:
            return
        # 检查有没有其他当前节点的请求
        for req in self.to_pickup_requests.copy():
            if req.from_node == self.current_node and self.can_pickup(req)[0]:
                self.pickup_request(req, consider_others=False)
        for req in self.to_delivery_requests.copy():
            if req.to_node == self.current_node:
                self.delivery_request(req, consider_others=False)

    def can_delivery(self, request):
        # 当前能否delivery请求
        travel_time = self.dist_matrix[self.current_node][request.to_node]
        delivery_time = self.available_time + travel_time
        cost = self.cost_matrix[self.current_node][request.to_node]

        if delivery_time <= self.time_limit:
            return True, travel_time, cost
        return False, travel_time, cost

    def delivery_request(self, request, consider_others=True):
        # 前往请求节点 delivery
        # 更新车辆状态
        self.available_time += self.dist_matrix[self.current_node][request.to_node]
        if self.current_node != request.to_node:
            self.route.append(request.to_node)
            self.current_node = request.to_node
        if self.available_time != self.times[-1]:
            self.times.append(self.available_time)
            self.load.append(self.load[-1] - 1)
        else:
            self.load[-1] -= 1
        self.to_delivery_requests.remove(request)

        # 更新请求状态
        request.delivered = True
        request.delivery_time = self.times[-1]
        request.serving_vehicle = self.vehicle_id

        # 放入已服务列表
        self.served_requests.append(request)

        if not consider_others:
            return
        # 检查有没有其他当前节点的请求
        for req in self.to_pickup_requests.copy():
            if req.from_node == self.current_node and self.can_pickup(req)[0]:
                self.pickup_request(req, consider_others=False)
        for req in self.to_delivery_requests.copy():
            if req.to_node == self.current_node:
                self.delivery_request(req, consider_others=False)

class NearestHeuristic:
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
        self.requests = []
        self.vehicles = []

    def initialize_requests_and_vehicles(self):
        for m in range(self.M):
            self.requests.append(Request(m, self.from_req[m], self.to_req[m], self.appear[m]))
        for k in range(self.K):
            self.vehicles.append(Vehicle(k, self.start[k], self.capacity[k], self.join_time[k], self.dist, self.cost, self.T))
            # 处理预装载的请求
            for m in range(self.M):
                if self.pre_load[k][m]:
                    self.vehicles[k].load[-1] += 1
                    self.vehicles[k].to_delivery_requests.append(self.requests[m])
                    self.requests[m].picked_up = True
                    self.requests[m].pickup_time = self.join_time[k]
                    self.requests[m].serving_vehicle = k

    def objective_function(self):
        total_value = 0
        total_cost = 0
        total_penalty = 0
        # 计算总价值
        for req in self.requests:
            if req.delivered:
                total_value += self.value[req.req_id]
        # 计算总花费
        for veh in self.vehicles:
            for i in range(len(veh.route)-1):
                total_cost += self.cost[veh.route[i]][veh.route[i+1]]
        # 计算总惩罚
        for req in self.requests:
            if not req.delivered:
                total_penalty += self.penalty[req.req_id]
        return total_value - total_cost - total_penalty

    def solve(self, show=True):
        self.initialize_requests_and_vehicles()
        # 按出现时间排序请求
        requests_sorted = sorted(self.requests, key=lambda x: x.appear_time)
        # 跳过预装载的请求
        for req in self.requests:
            if req.picked_up:
                requests_sorted.remove(req)

        # 按时间顺序，分配pickup和delivery
        for t in range(self.T + 1):
            requests_sorted_removing = []
            for req in requests_sorted:
                if req.appear_time <= t:
                    # 尝试分配给代价最小的可接取请求的车辆
                    min_cost = np.inf
                    selected_vehicle = None
                    selected_travel_time = 0
                    for veh in self.vehicles:
                        if t < veh.join_time:
                            continue
                        can_pick, travel_time, cost = veh.can_pickup(req)
                        if can_pick:
                            if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                                min_cost = cost
                                selected_vehicle = veh
                                selected_travel_time = travel_time
                    if selected_vehicle:
                        selected_vehicle.to_pickup_requests.append(req)
                        requests_sorted_removing.append(req)
            for req in requests_sorted_removing:
                requests_sorted.remove(req)

            # 为空闲的车选择下一步行动
            for veh in self.vehicles:
                # 下面这步是干啥的？
                # 检查待pickup请求是不是可以分配给其他车辆，来减小cost
                # for req in veh.to_pickup_requests:
                #     min_cost = np.inf
                #     selected_vehicle = None
                #     selected_travel_time = 0
                #     for other_veh in self.vehicles:
                #         if t < other_veh.join_time:
                #             continue
                #         can_pick, travel_time, cost = other_veh.can_pickup(req)
                #         if can_pick:
                #             if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                #                 min_cost = cost
                #                 selected_vehicle = other_veh
                #                 selected_travel_time = travel_time
                #     if selected_vehicle:
                #         selected_vehicle.to_pickup_requests.append(req)
                #         veh.to_pickup_requests.remove(req)

                if t < veh.available_time:
                    continue
                # 检查是否有请求需要delivery，选择cost最小的请求
                min_cost = np.inf
                selected_request = None
                selected_travel_time = 0
                for req in veh.to_delivery_requests:
                    can_delivery, travel_time, cost = veh.can_delivery(req)
                    if can_delivery:
                        if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                            min_cost = cost
                            selected_request = req
                            selected_travel_time = travel_time
                if selected_request:
                    veh.delivery_request(selected_request)

                # 检查是否有请求需要pickup，选择cost最小的请求
                min_cost = np.inf
                selected_request = None
                selected_travel_time = 0
                for req in veh.to_pickup_requests:
                    can_pickup, travel_time, cost = veh.can_pickup(req)
                    if can_pickup:
                        if cost < min_cost or (cost == min_cost and travel_time < selected_travel_time):
                            min_cost = cost
                            selected_request = req
                            selected_travel_time = travel_time
                if selected_request:
                    veh.pickup_request(selected_request)
            
        # 计算 objective function
        objective_value = self.objective_function()

        _pickup = np.zeros((self.K, self.M, self.T + 1), dtype=bool)
        _location = np.full((self.K, self.T + 1), -1, dtype=int)

        for req in self.requests:
            if req.picked_up:
                if req.appear_time != -1:
                    _pickup[req.serving_vehicle, req.req_id, req.pickup_time] = 1

        for veh in self.vehicles:
            for t, loc in zip(veh.times, veh.route):
                _location[veh.vehicle_id, t] = loc

        if not show:
            return _pickup, _location

        # Print results
        print("Objective Value:", objective_value)
        print("=======================Request Result=======================")
        for req in self.requests:
            print(f"Request {req.req_id} From: {req.from_node} to: {req.to_node}, Appear: {req.appear_time}, Value: {self.value[req.req_id]}, Penalty: {self.penalty[req.req_id]}")
            if req.picked_up:
                print(f"Car{req.serving_vehicle} picked up at t={req.pickup_time}")
            if req.delivered:
                print(f"Car{req.serving_vehicle} delivered at t={req.delivery_time}")
        print("=======================Vehicle Result=======================")
        for veh in self.vehicles:
            print(f"Car {veh.vehicle_id} Route:")
            for i, time in enumerate(veh.times):
                pick_up = [req.req_id for req in self.requests if req.serving_vehicle == veh.vehicle_id and req.pickup_time == time]
                delivery = [req.req_id for req in self.requests if req.serving_vehicle == veh.vehicle_id and req.delivery_time == time]
                print(f"t = {time}, location = {veh.route[i]}, load = {veh.load[i]}, pick up = {pick_up}, delivery = {delivery}")
        
        return _pickup, _location

if __name__ == "__main__":
    # 示例数据
    N = 5
    M = 6
    K = 2
    T = 12
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

    # 运行启发式算法
    heuristic = NearestHeuristic(N, M, K, T, start, capacity, join_time, dist, cost, from_req, to_req, appear, value, penalty, pre_load)
    heuristic.solve()
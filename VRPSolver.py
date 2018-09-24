import re
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.cluster import KMeans


class VRPSolver:
    def __init__(self, instance, n_clusters=10, n_jobs=1):
        self.name, self.vehicle_number, self.vehicle_capacity, columns_names, lines = self.read_data(instance)
        self.df = pd.DataFrame(data=lines, columns=columns_names, dtype=int).set_index('cust_no')
        self.distances = None
        self.n_jobs = n_jobs
        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
            self.distances = parallel(joblib.delayed(self._dist_from_to)(i, j)
                                      for i in range(self.df.shape[0]) for j in range(i))
        assert self.distances is not None, 'Lul wat'
        self.distances = dict(self.distances)
        self.current_route = list()
        self.arrival = np.zeros(self.df.shape[0])
        self.solution = list()
        self.opened = self.df.loc[:, 'ready_time']
        self.closed = self.df.loc[:, 'due_date']
        self.service = self.df.loc[:, 'service_time']
        self.demand = self.df.loc[:, 'demand']
        self.current_cluster = None

        kmc = KMeans(n_clusters, n_jobs=self.n_jobs)
        coords = self.df.loc[:, ['xcoord', 'ycoord']]
        self.df['cluster'] = kmc.fit_predict(coords)

    def read_data(self, filepath):
        with open(filepath, 'r') as f:
            name = f.readline()
            f.readline()
            f.readline()
            f.readline()
            vehicle_number, vehicle_capacity = re.sub('\s\s+', ';', f.readline().strip()).split(';')
            f.readline()
            f.readline()
            columns_names = re.sub('\s\s+', ';', f.readline().strip()).lower().split(';')
            columns_names = [re.sub(' ', '_', name.strip('.')) for name in columns_names]
            f.readline()
            lines = f.readlines()
            lines = [re.sub('\s\s+', ';', line.strip()).split(';') for line in lines]
            return name, int(vehicle_number), int(vehicle_capacity), columns_names, lines

    def _dist_from_to(self, client_i, client_j):
        return ((client_i, client_j), np.linalg.norm(
            self.df.loc[client_i, ['xcoord', 'ycoord']].values
            - self.df.loc[client_j, ['xcoord', 'ycoord']].values
        ))

    def dist(self, client_i, client_j):
        if client_i == client_j:
            return 0.0
        if client_i < client_j:
            return self.distances[(client_j, client_i)]
        else:
            return self.distances[(client_i, client_j)]

    def _unload_and_move(self, i, j):
        return self.arrival[i] + self.service[i] + self.dist(i, j)

    def _c1(self, u, i, j):
        return self.dist(i, u) + self.dist(u, j) - self.dist(i, j)

    def _c2(self, u, i, j):
        return self.closed[j] - (self._unload_and_move(i, j)) \
               - (self.closed[j] - (self._possible_arrival(u, i) + self.service[u] + self.dist(u, j)))

    def _c3(self, u, i, j):
        return self.closed[u] - (self._unload_and_move(i, u))

    def _local_disturbance(self, u, i, j):
        b1 = b2 = b3 = 1 / 3
        c1 = self._c1(u, i, j)
        c2 = self._c2(u, i, j)
        c3 = self._c3(u, i, j)
        return b1 * c1 + b2 * c2 + b3 * c3

    def _possible_arrival(self, u, i):
        """Time of arrival after possible insertion `u` after `i`"""
        return np.max([self._unload_and_move(i, u), self.opened[u]])

    def _is_feasible_insertion(self, u, i, j):
        arrival_u = self._possible_arrival(u, i)
        is_feasible = arrival_u + self.service[u] + self.dist(u, j) < self.closed[j]
        return is_feasible

    def _global_disturbance(self, u, feasible_insertion_points):
        local_disturbances = [(i, self._local_disturbance(u, i, j)) for i, j in feasible_insertion_points]
        insertion_point = local_disturbances[np.argmin(local_disturbances, axis=0)[1]][0]
        mean_disturbance = np.sum(local_disturbances, axis=0)[1] / len(local_disturbances)
        return mean_disturbance, insertion_point

    def _accessibility(self, u, feasible_insertion_points):
        return 1 / self._global_disturbance(u, feasible_insertion_points)

    def _internal_impact(self, u, feasible_insertion_points):
        return self._global_disturbance(u, feasible_insertion_points)

    def _own_impact(self, u, insertion_point):
        #return self.arrival[u] - self.opened[u]
        return 0.0

    def _external_impact(self, u):
        if self.current_cluster.shape[0] == 1:
            return 0.0
        non_routed_points = self.current_cluster.query('index != @u')
        external_impact = (1 / non_routed_points.shape[0]) * non_routed_points.apply(
            lambda x: np.max([
                x.due_date - self.current_cluster.loc[u, 'ready_time'] - self.dist(u, x.name),
                self.current_cluster.loc[u, 'due_date'] - x.ready_time - self.dist(u, x.name),
            ]), axis=1).sum()
        return external_impact

    def _impact(self, u, possible_insertions):
        bo = 0.0
        bi = be = 1 / 2
        feasible_insertion_points = [(i, j) for i, j in possible_insertions
                                     if self._is_feasible_insertion(u, i, j)]
        if len(feasible_insertion_points) == 0:
            return float('inf'), 0

        internal_impact, insertion_point = self._internal_impact(u, feasible_insertion_points)
        weighted_internal_impact = bi * internal_impact
        weighted_own_impact = bo * self._own_impact(u, insertion_point)
        weighted_external_impact = be * self._external_impact(u)
        return weighted_internal_impact + weighted_own_impact + weighted_external_impact, insertion_point

    def _cost_function(self, route):
        return np.sum(self.dist(client, route[i + 1]) for i, client in enumerate(route[:-1]))

    def _check_route(self, route):
        assert route[0] == 0, "Route doesn't start at depot"
        assert route[-1] == 0, "Route doesn't end in depot"
        is_not_overloaded = np.sum((self.demand[i] for i in route)) <= self.vehicle_capacity
        return is_not_overloaded

    def _pick_seed_customer(self, candidates):
        return candidates.ready_time.idxmin()

    def _insert_client(self, client, insertion_point):
        prev_client = 0
        for i, c in enumerate(self.current_route):
            if c == insertion_point:
                self.current_route.insert(i + 1, client)
                prev_client = c
                break
        self.arrival[client] = np.max([self._unload_and_move(prev_client, client), self.opened[client]])

    def _route_cost(self, route):
        roads = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(tuple)([customer, route[i + 1]]) for i, customer in enumerate(route[:-1])
        )
        return sum((self.dist(i, j) for i, j in roads))

    def solution_cost(self, sln):
        return sum(map(self._route_cost, sln))

    def _each_customer_is_served(self, sln):
        from functools import reduce
        return len(reduce(lambda x, y: x.union(y), map(set, sln))) == self.df.shape[0]

    def _each_vehicle_is_not_overloaded(self, sln):
        def vehicle_is_not_overloaded(clients):
            return self.demand[clients].sum() <= self.vehicle_capacity
        return all(map(vehicle_is_not_overloaded, sln))

    def _vehicle_count_is_lesser_than_available(self, sln):
        return len(sln) <= self.vehicle_number

    def _all_unloads_in_correct_time_window(self, sln):
        return all([self.opened[c] <= self.arrival[c] < self.closed[c] for route in sln for c in route[:-1]])

    def _vehicles_is_not_late_in_depot(self, sln):
        return all([self.opened[route[-1]]
                    <= self.arrival[route[-2]] + self.service[route[-2]] + self.dist(route[-2], route[-1])
                    < self.closed[route[-1]] for route in sln])

    def is_solution_feasible(self, sln):
        is_feasible = True
        each_customer_is_served = self._each_customer_is_served(sln)
        if not each_customer_is_served:
            is_feasible = False
            print("each_customer_is_served = False")
        each_vehicle_is_not_overloaded = self._each_vehicle_is_not_overloaded(sln)
        if not each_vehicle_is_not_overloaded:
            is_feasible = False
            print("each_vehicle_is_not_overloaded = False")
        vehicle_count_is_lesser_than_available = self._vehicle_count_is_lesser_than_available(sln)
        if not vehicle_count_is_lesser_than_available:
            is_feasible = False
            print("vehicle_count_is_lesser_than_available = False")
        all_unloads_in_correct_time_window = self._all_unloads_in_correct_time_window(sln)
        if not all_unloads_in_correct_time_window:
            is_feasible = False
            print("all_unloads_in_correct_time_window = False")
        vehicles_is_not_late_in_depot = self._vehicles_is_not_late_in_depot(sln)
        if not vehicles_is_not_late_in_depot:
            is_feasible = False
            print("vehicles_is_not_late_in_depot = False")
        return is_feasible

    def get_initial_solution(self):
        def tupled(y, func, kwargs):
            return tuple([y, func(*kwargs)])

        for cluster_i in self.df.cluster.unique():
            self.current_route.append(0)
            self.current_cluster = self.df.query('cluster == @cluster_i and index != "0"')
            seed = self._pick_seed_customer(self.current_cluster)
            self.current_route.append(seed)
            self.arrival[seed] = np.max([self.dist(0, seed), self.opened[seed]])
            self.current_route.append(0)
            self.current_cluster.query('index != @seed', inplace=True)
            with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
                while self.current_cluster.shape[0] != 0 and self._check_route(self.current_route):
                    insertion_points = parallel(
                        joblib.delayed(tuple)([customer, self.current_route[i + 1]]) for i, customer in
                        enumerate(self.current_route[:-1])
                    )
                    insertion_candidates_impact = parallel(
                        joblib.delayed(tupled)(candidate, self._impact, (candidate, insertion_points))
                        for candidate in self.current_cluster.index
                    )
                    chosen_one, (_, insertion_point) = min(insertion_candidates_impact, key=lambda x: x[1][0])
                    self._insert_client(chosen_one, insertion_point)
                    self.current_cluster.query('index != @chosen_one', inplace=True)
            self.solution.append(self.current_route)
            self.current_cluster = None
            self.current_route = list()
        return len(self.solution), self.solution, self.solution_cost(self.solution)


if __name__ == "__main__":
    instances = glob.glob('./instances/*.txt')
    slv = VRPSolver(instances[0], n_jobs=8)
    n_vehicles, solution, cost = slv.get_initial_solution()
    print(n_vehicles, solution, cost, slv.is_solution_feasible(solution))

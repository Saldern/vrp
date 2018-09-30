import re
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.cluster import KMeans


class VRPSolver:
    def __init__(self, instance, n_clusters=None, n_jobs=1):
        self.name, self.vehicle_number, self.vehicle_capacity, columns_names, lines = self.read_data(instance)
        self.n_clusters = n_clusters
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
        self.routed_clients = list()

        self.df['cluster'] = 0
        coords = self.df.loc[:, ['xcoord', 'ycoord']]
        if self.n_clusters is None:
            self.n_clusters = 8
            while self.df.groupby('cluster').demand.sum().max() > self.vehicle_capacity:
                self.n_clusters += 3
                kmc = KMeans(self.n_clusters, n_jobs=self.n_jobs)
                self.df['cluster'] = kmc.fit_predict(coords)
            while self.df.groupby('cluster').demand.sum().max() <= self.vehicle_capacity:
                self.n_clusters -= 1
                kmc = KMeans(self.n_clusters, n_jobs=self.n_jobs)
                self.df['cluster'] = kmc.fit_predict(coords)
            self.n_clusters += 1
        kmc = KMeans(self.n_clusters, n_jobs=self.n_jobs)
        self.df['cluster'] = kmc.fit_predict(coords)
        self.df.loc[0, 'cluster'] = self.n_clusters

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
        if self._current_demand() + self.demand[u] > self.vehicle_capacity:
            return False

        arrival_u = self._possible_arrival(u, i)
        late_arrival = arrival_u >= self.closed[u]
        if late_arrival:
            return False

        insertion_index = 0
        for idx, client in enumerate(self.current_route):
            if client == i:
                insertion_index = idx
                break

        next_arrival = np.max([arrival_u + self.service[u] + self.dist(u, j), self.opened[j]])
        if next_arrival >= self.closed[j]:
            return False

        rest_of_the_route = self.current_route[insertion_index+1:]
        for idx, client in enumerate(rest_of_the_route[:-1]):
            next_client = rest_of_the_route[idx+1]
            next_arrival = np.max([next_arrival + self.service[client] + self.dist(client, next_client),
                                   self.opened[next_client]])
            if next_arrival >= self.closed[next_client]:
                return False
        return True

    def _global_disturbance(self, u, feasible_insertion_points, verbose):
        local_disturbances = [(i, self._local_disturbance(u, i, j)) for i, j in feasible_insertion_points]
        if verbose > 0:
            print("===== Local disturbance =====")
            for i, j in local_disturbances:
                print('    after {} = {}'.format(i, j))
        insertion_point = local_disturbances[np.argmin(local_disturbances, axis=0)[1]][0]
        mean_disturbance = np.sum(local_disturbances, axis=0)[1] / len(local_disturbances)
        if verbose > 0:
            print('Minimal disturbance is {:.2f} at {}'.format(np.min(local_disturbances, axis=0)[1], insertion_point))
        return mean_disturbance, insertion_point

    def _internal_impact(self, u, feasible_insertion_points, verbose):
        return self._global_disturbance(u, feasible_insertion_points, verbose)

    def _own_impact(self, u, insertion_point):
        own_impact = self._unload_and_move(insertion_point, u) - self.opened[u]
        return own_impact

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

    def _impact(self, u, possible_insertions, verbose):
        bo = 0.33
        bi = 0.33
        be = 0.33
        feasible_insertion_points = [(i, j) for i, j in possible_insertions
                                     if self._is_feasible_insertion(u, i, j)]
        if verbose > 0:
            print('Client {}'.format(u))
            print("===== Feasible insertions =====")
            for i, j in feasible_insertion_points:
                print('    between {} and {}'. format(i, j))
        if len(feasible_insertion_points) == 0:
            return float('inf'), -1

        internal_impact, insertion_point = self._internal_impact(u, feasible_insertion_points, verbose)
        weighted_internal_impact = bi * internal_impact
        weighted_own_impact = bo * self._own_impact(u, insertion_point)
        weighted_external_impact = be * self._external_impact(u)
        return weighted_internal_impact + weighted_own_impact + weighted_external_impact, insertion_point

    def _cost_function(self, route):
        return np.sum(self.dist(client, route[i + 1]) for i, client in enumerate(route[:-1]))

    def _current_demand(self):
        return np.sum([self.demand[i] for i in self.current_route])

    def _pick_seed_customer(self, candidates):
        return candidates.ready_time.idxmin()

    def recount_arrivals(self, solution):
        for route in solution:
            self.arrival[0] = 0.0
            for i, client in enumerate(route[:-2]):
                next_client = route[i+1]
                self.arrival[next_client] = np.max([self._unload_and_move(client, next_client),
                                                    self.opened[next_client]])


    def _insert_client(self, client, insertion_point):
        self.routed_clients.append(client)
        insertion_index = 0
        for i, c in enumerate(self.current_route):
            if c == insertion_point:
                insertion_index = i
                self.current_route.insert(insertion_index + 1, client)
                break
        for i, client in enumerate(self.current_route[insertion_index:-2]):
            next_client = self.current_route[insertion_index:][i+1]
            self.arrival[next_client] = np.max([self._unload_and_move(client, next_client), self.opened[next_client]])

    def route_cost(self, route):
        roads = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(tuple)([customer, route[i + 1]]) for i, customer in enumerate(route[:-1])
        )
        return sum((self.dist(i, j) for i, j in roads))

    def solution_cost(self, sln):
        return sum(map(self.route_cost, sln))

    def _each_customer_is_served(self, sln):
        from functools import reduce
        return len(reduce(lambda x, y: x.union(y), map(set, sln))) == self.df.shape[0]

    def vehicle_is_not_overloaded(self, clients):
        return self.demand[clients].sum() <= self.vehicle_capacity

    def _each_vehicle_is_not_overloaded(self, sln):
        return all(map(self.vehicle_is_not_overloaded, sln))

    def _vehicle_count_is_lesser_than_available(self, sln):
        return len(sln) <= self.vehicle_number

    def check_time_windows(self, route):
        return all([self.opened[c] <= self.arrival[c] < self.closed[c] for c in route[:-1]])

    def check_late_to_depot(self, route):
        depot = route[-1]
        last_client = route[-2]
        return (self.opened[depot] <= self.arrival[last_client] + self.service[last_client]
                + self.dist(last_client, depot) < self.closed[depot])

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
            #print("each_customer_is_served = False")
        each_vehicle_is_not_overloaded = self._each_vehicle_is_not_overloaded(sln)
        if not each_vehicle_is_not_overloaded:
            is_feasible = False
            #print("each_vehicle_is_not_overloaded = False")
        vehicle_count_is_lesser_than_available = self._vehicle_count_is_lesser_than_available(sln)
        if not vehicle_count_is_lesser_than_available:
            is_feasible = False
            #print("vehicle_count_is_lesser_than_available = False")
        all_unloads_in_correct_time_window = self._all_unloads_in_correct_time_window(sln)
        if not all_unloads_in_correct_time_window:
            is_feasible = False
            #print("all_unloads_in_correct_time_window = False")
        vehicles_is_not_late_in_depot = self._vehicles_is_not_late_in_depot(sln)
        if not vehicles_is_not_late_in_depot:
            is_feasible = False
            #print("vehicles_is_not_late_in_depot = False")
        return is_feasible

    def _init_route(self, cluster_i):
        self.current_route.append(0)
        seed = self._pick_seed_customer(self.current_cluster)
        self.routed_clients.append(seed)
        self.current_route.append(seed)
        self.arrival[seed] = np.max([self.dist(0, seed), self.opened[seed]])
        self.current_route.append(0)
        self.current_cluster.query('index != @seed', inplace=True)

    def _close_route(self):
        # print("add route", self.current_route)
        self.solution.append(self.current_route)
        self.current_route = list()

    def get_initial_solution(self, verbose=0):
        def tupled(y, func, kwargs):
            return tuple([y, func(*kwargs)])

        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
            for cluster_i in self.df.cluster.unique():
                if cluster_i == self.df.cluster.max():
                    self.routed_clients.append(0)
                    continue
                routed_clients = self.routed_clients
                self.current_cluster = self.df.query(
                    'cluster == @cluster_i and index != "0" and index not in @routed_clients')
                while self.current_cluster.shape[0] != 0:
                    self._init_route(cluster_i)
                    insertion_points = parallel(
                        joblib.delayed(tuple)([customer, self.current_route[i + 1]]) for i, customer in
                        enumerate(self.current_route[:-1])
                    )
                    insertion_candidates_impact = parallel(
                        joblib.delayed(tupled)(candidate, self._impact, (candidate, insertion_points, verbose))
                        for candidate in self.current_cluster.index
                    )
                    # non parallel version
                    # insertion_candidates_impact = [
                    #     (candidate, self._impact(candidate, insertion_points, verbose))
                    #     for candidate in self.current_cluster.index
                    # ]
                    insertion_candidates_impact = [(c, i, point) for c, (i, point)
                                                   in insertion_candidates_impact if point != -1]
                    while len(insertion_candidates_impact) != 0:
                        if verbose > 0:
                            print('===== Candidates: =====')
                            for (client, impact, insert_after) \
                                    in sorted(insertion_candidates_impact, key=lambda x: x[1]):
                                print('Client {} (openned: {}, closed: {}), impact={}, insert after {}'.format(
                                    client, self.opened[client], self.closed[client], impact, insert_after))
                            print()
                            print('===== Current route: =====')
                            print('->'.join([str(i) for i in self.current_route]))
                            print('===== Arrival =====')
                            print('->'.join(['{:.1f}'.format(self.arrival[i]) for i in self.current_route]))
                            print('===== Closed =====')
                            print('->'.join(['{:.0f}'.format(self.closed[i]) for i in self.current_route]))

                        chosen_one, _, insertion_point = min(insertion_candidates_impact, key=lambda x: x[1])
                        self._insert_client(chosen_one, insertion_point)
                        self.current_cluster.query('index != @chosen_one', inplace=True)
                        if verbose > 0:
                            print()
                            print('===== Route after insert: =====')
                            print('->'.join([str(i) for i in self.current_route]))
                            print('===== Arrival =====')
                            print('->'.join(['{:.1f}'.format(self.arrival[i]) for i in self.current_route]))
                            print('===== Closed =====')
                            print('->'.join(['{:.0f}'.format(self.closed[i]) for i in self.current_route]))
                        insertion_points = parallel(
                            joblib.delayed(tuple)([customer, self.current_route[i + 1]]) for i, customer in
                            enumerate(self.current_route[:-1])
                        )
                        insertion_candidates_impact = parallel(
                            joblib.delayed(tupled)(candidate, self._impact, (candidate, insertion_points, verbose))
                            for candidate in self.current_cluster.index
                        )
                        insertion_candidates_impact = [(c, i, point) for c, (i, point)
                                                       in insertion_candidates_impact if point != -1]
                    self._close_route()

            return len(self.solution), self.solution, self.solution_cost(self.solution)


if __name__ == "__main__":
    instances = glob.glob('./instances/*.txt')
    slv = VRPSolver(instances[0], n_jobs=8)
    n_vehicles, solution, cost = slv.get_initial_solution(0)
    print(n_vehicles, cost, slv.is_solution_feasible(solution))

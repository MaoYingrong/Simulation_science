import numpy as np
import itertools
from copy import deepcopy


class NKModel:
    def __init__(self, N, K, R, G, U, prob_jump=1, Beta=0.1):
        self.N = N  # Number of components
        self.K = K  # Number of interactions
        self.R = R  # Range; Number of components to change
        self.G = G  # Granularity; The smaller the value, the more precise the representation
        self.U = U  # Inaccuracy; The standard deviation of the noise added to the predicted value
        self.Beta = Beta  # The coefficient of cost of changing the state
        self.environment = self.generate_environment()
        self.landscape = self.generate_landscape()

    def generate_environment(self):
        """Generate the coefficients for the NK landscape"""
        landscapes = np.random.randn(self.N, 2**(self.K+1))
        return landscapes

    def get_fitness(self, state):
        """Calculate the fitness of a state"""
        total_fitness = 0
        for i in range(self.N):
            # Determine the index for the sub-landscape
            index = state[i]
            for j in range(1, self.K+1):
                interaction_partner = (i + j) % self.N
                index = (index << 1) | state[interaction_partner]
            total_fitness += self.environment[i, index]
        return total_fitness / self.N

    def generate_landscape(self):
        """Generate the actual values of the entire landscape"""
        landscape = {}
        for i in range(2**self.N):
            state = [int(x) for x in f"{i:0{self.N}b}"]
            landscape[tuple(state)] = self.get_fitness(state)
        return landscape

    def generate_changed_series(self, series, num_change):
        """Generate all possible series that can be obtained by changing a specified number of elements in the series"""
        series_list = list(series)
        length = len(series_list)
        index_combinations = itertools.combinations(range(length), num_change)

        changed_series = []
        for indices in index_combinations:
            new_series = series_list[:]
            for index in indices:
                new_series[index] = 1 if new_series[index] == 0 else 0
            changed_series.append(new_series)
        
        return changed_series


    def separate_list_of_lists(self, lst, n_parts):
        """Separate a list of lists into n parts"""
        part_size = len(lst) // n_parts
        remainder = len(lst) % n_parts

        parts = []
        start = 0
        for i in range(n_parts):
            end = start + part_size + (1 if i < remainder else 0)
            temp = deepcopy(lst[start:end])
            for i in range(len(temp)):
                temp[i] = [int(x) for x in temp[i]]
            parts.append(lst[start:end])
            start = end
        
        return parts


    def generate_representation(self, state):
        """
        Generate a representation of the fitness landscape that can be used to predict the performance of areas within the range
        """
        representation = {}
        total_value = 0
        num_separate = 0
        if self.R > len(state):
            self.R = len(state)
        for i in range(0, self.R+1):
            if i <= len(state) / 2:
                num_separate =  len(self.generate_changed_series(state, i // self.G))
            else:
                num_separate = len(self.generate_changed_series(state, (len(state) - i) // self.G))
            
            lst = self.generate_changed_series(state, i)
            separated_parts = self.separate_list_of_lists(lst, num_separate)
            for part in separated_parts:
                for j in part:
                    representation[tuple(j)] = {"cost":i}
                    total_value += self.landscape[tuple(j)]
                predicted_value = total_value / len(part)
                for k in part:
                    representation[tuple(k)]["value"] = predicted_value
                total_value = 0
        self.representation = representation
        return representation


    def predict_performance(self, state):
        """
        Based on the representation generated and the inaccuracy, predict the performance of the states within the range
        """
        representation_dic = self.generate_representation(state)
        predicted_dict = deepcopy(representation_dic)
        for key in predicted_dict:
            predicted_dict[key]["value"] = predicted_dict[key]["value"] * (1 + np.random.normal(0, self.U))
        return predicted_dict
    
    
    def move_agent(self, current_state):
        """calculate the best move for the agent based on the predicted performance of the states within the range"""
        best_move = current_state[:]
        max_objective = float('-inf')

        potential_states = self.predict_performance(current_state)

        for state_key, state_value in potential_states.items():
            objective = state_value["value"] - self.Beta * state_value["cost"]

            if objective > max_objective:
                max_objective = objective
                best_move = state_key
                cost = state_value["cost"]
                predicted_value = state_value["value"]

        return best_move, predicted_value, cost * self.Beta


    def simulate(self, iterations=10):
        """Simulate the agent moving through the landscape"""
        # Random initial state
        current_state = tuple(np.random.randint(0, 2, self.N))
        results = []
        self.initial_state = current_state

        for _ in range(iterations):
            new_state, predicted_value, cost = self.move_agent(current_state)
            actual_performance = self.get_fitness(new_state)
            result = actual_performance - self.Beta * cost
            results.append(result)
            current_state = new_state

        return np.mean(results)
    

    def simulate_print(self, iterations=10):
        """simulation with print statements"""
        current_state = tuple(np.random.randint(0, 2, self.N))
        results = []
        self.initial_state = current_state

        for _ in range(iterations):
            new_state, predicted_value, cost = self.move_agent(current_state)
            actual_performance = self.get_fitness(new_state)
            result = actual_performance - self.Beta * cost
            results.append(result)
            print(f"Current State: {current_state}\nNew State: {new_state}\nCost: {cost}\nPredicted Value: {predicted_value}\nActual Performance: {actual_performance}\nResult: {result}")
            current_state = new_state

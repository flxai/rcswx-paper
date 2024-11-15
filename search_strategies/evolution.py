from collections import deque
from os.path import join, exists
from os import makedirs
import pickle
import time
import random

from tqdm import tqdm

from search_strategies.random_search import Sampler
from visualise import visualise_derivation_tree
from search_state import DerivationTreeNode, Stack
from pcfg import OutOfOptionsError
from utils import Timer


class Individual(object):
    """A class representing a model containing an architecture, its modules and its accuracy."""

    def __init__(
        self,
        id,
        parent_id,
        root=None,
        accuracy=None,
        age=0,
        hpo_dict=None,
    ):
        self.id = id
        self.parent_id = parent_id
        self.root = root
        self.accuracy = accuracy
        self.age = age
        self.hpo_dict = hpo_dict

        self.alive = True

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Individual(id={self.id}, accuracy={self.accuracy}, age={self.age}"


class Population(deque):
    """A class representing a population of models."""

    def __init__(self, individuals):
        self.individuals = individuals

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Population(individuals={self.individuals})"

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __setitem__(self, idx, value):
        self.individuals[idx] = value

    def append(self, individual):
        self.individuals.append(individual)

    def popleft(self):
        individual = self.individuals.pop(0)
        individual.alive = False

    def max(self, key):
        return max(self.individuals, key=key)

    def sample(self, k):
        return random.choice(self.individuals, k=k)

    def extend(self, individuals):
        self.individuals.extend(individuals)

    def sort(self, key):
        self.individuals.sort(key=key)

    def __iter__(self):
        return iter(self.individuals)

    def __next__(self):
        return next(self.individuals)

    def __contains__(self, item):
        return item in self.individuals

    def index(self, item):
        return self.individuals.index(item)

    def remove(self, item):
        self.individuals.remove(item)

    def tournament_selection(self, k, key):
        sample = []
        while len(sample) < k:
            candidate = random.choice(list(self.individuals))
            sample.append(candidate)
        return max(sample, key=key)

    def age(self):
        for individual in self.individuals:
            individual.age += 1

    def tolist(self):
        return self.individuals


class Evolver(Sampler):
    def __init__(
        self,
        mutation_strategy="random",
        mutation_rate=0.1,
        crossover_strategy="one_point",
        crossover_rate=0.5,
        selection_strategy="tournament",
        tournament_size=10,
        elitism=True,
        pcfg,
        mode,
        max_depth=20,
        time_limit=300,
        max_id_limit=1000,
        verbose=False,
    ):
        super().__init__(
            pcfg=pcfg,
            mode=mode,
            max_depth=max_depth,
            time_limit=time_limit,
            max_id_limit=max_id_limit,
            verbose=verbose
        )
        self.mutation_strategy = mutation_strategy
        self.mutation_rate = mutation_rate
        self.crossover_strategy = crossover_strategy
        self.crossover_rate = crossover_rate
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.elitism = elitism

    def __call__(self, population):
        # select the parents
        parent1 = self.select(population)
        parent2 = self.select(population)
        # crossover the parents
        child = self.crossover(parent1, parent2)
        # mutate the child
        child = self.mutate(child)
        return child

    def select(self, population):
        if self.selection_strategy == "tournament":
            return population.tournament_selection(self.tournament_size, key=lambda x: x.accuracy)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.crossover_strategy == "one_point":
                return self.one_point_crossover(parent1, parent2)
            elif self.crossover_strategy == "two_point":
                return self.two_point_crossover(parent1, parent2)
        return parent1

    def one_point_crossover(self, parent1, parent2):
        # select a random node from parent1
        node1 = random.choice(parent1.root.serialise())
        # select a random node from parent2
        node2 = random.choice(parent2.root.serialise())
        # create a new individual by swapping the subtrees
        child = parent1.root.copy()
        child.replace(node1, node2)
        return child

    def two_point_crossover(self, parent1, parent2):
        # select a random node from parent1
        node1 = random.choice(parent1.root.serialise())
        # select a random node from parent2
        node2 = random.choice(parent2.root.serialise())
        # create a new individual by swapping the subtrees
        child = parent1.root.copy()
        child.replace(node1, node2)
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            if self.mutantion_strategy == "random":
                return self.random_mutation(individual)
        return individual

    def random_mutation(self, individual):
        # choose a random node to mutate
        node = random.choice(individual.root.serialise())
        # mutate the node
        success = False
        while not success:
            try:
                root = self.mutate_node(node)
                individual.root = root
                success = True
            except Exception as e:
                print("MutationError:", e)
        return individual

    def mutate_node(self, node):
        # sample a new subtree rooted at this node
        new_node = self.__call__(node) # run sampler
        # test to see if the new architecture is valid
        new_node.get_root().validate() # raises an exception if the architecture is invalid
        return new_node.get_root()


class Evolution:
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            max_id_limit=1000,
            time_limit=300,
            max_depth=20,
            verbose=False,
            verbose_after_iteration=None,
            visualise=False,
            visualise_after_iteration=None,
            visualise_scale=0.5,
            figures_path=None,
            results_path=None,
            continue_search=False,
            # evolution specific parameters
            regularised=True, # use regularised evolution
            population_size=100, # number of individuals in the population
            mutation_strategy="random", # "random" or "gaussian"
            mutation_rate=0.1, # probability of mutation
            crossover_strategy="one_point", # "one_point" or "two_point"
            crossover_rate=0.5, # probability of crossover
            selection_strategy="tournament", # "tournament" or "roulette"
            tournament_size=10, # only used if selection_strategy is "tournament"
            elitism=True, # keep the best individual in the population
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.max_id_limit = max_id_limit
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.verbose = verbose
        self.verbose_after_iteration = verbose_after_iteration
        self.visualise = visualise
        self.visualise_after_iteration = visualise_after_iteration
        self.visualise_scale = visualise_scale
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search
        # evolution specific parameters
        self.regularised = regularised
        self.population_size = population_size

        self.sampler = Sampler(
            pcfg=self.pcfg,
            mode=self.mode,
            max_depth=self.max_depth,
            time_limit=self.time_limit,
            max_id_limit=self.max_id_limit,
            verbose=self.verbose
        )

        self.evolver = Evolver(
            mutation_strategy=mutation_strategy,
            mutation_rate=mutation_rate,
            crossover_strategy=crossover_strategy,
            crossover_rate=crossover_rate,
            selection_strategy=selection_strategy,
            tournament_size=tournament_size,
            elitism=elitism,
        )

        self.rewards = []
        self.iteration = 0
        self.population = []

        self.set_rng_state(seed=self.seed)

        if self.continue_search:
            self.load_results()

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps):
        print("-------------")
        print("Evolution")
        print(f"Steps: {steps}")
        print("--------------")

        # populate the first generation
        for i in tqdm(range(len(self.population), self.population_size), desc="Initialising population"):
            self.step(i, "sample")

        for i in tqdm(range(len(self.population), self.population_size), desc="Initialising population"):
            self.step(i, "evolve")

    def step(self, iteration, mode):
        global timer

        # sample a new individual
        timer = Timer()
        if mode == "sample":
            root = self.sampler(self.input_params)
        elif mode == "evolve":
            root = self.evolver(self.population)
        sample_duration = timer()

        # evaluate the network
        timer = Timer()
        reward = self.evaluation_fn(root)
        eval_duration = timer()

        # add the new individual to the population
        self.rewards.append((root.serialise(), reward, sample_duration, eval_duration))
        self.population.append(Individual(id=iteration, parent_id=None, root=root, accuracy=reward))
        print(f"Iteration {iteration}, reward: {reward}, sample duration: {sample_duration}, eval duration: {eval_duration}")
        print(f"Architecture:")
        for line in root.serialise():
            print(line)

        # visualise the derivation tree
        visualise_derivation_tree(
            root,
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            show=self.visualise,
        )

        # save the results
        self.save_results(iteration)

        timer.stop()

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            with open(join(self.results_path, f"search_results.pkl"), "wb") as f:
                pickle.dump({
                    "rewards": self.rewards,
                    "iteration": iteration,
                    "population": self.population,
                    "rng_state": random.getstate(),
                }, f)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        if exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                self.population = data["population"]
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")

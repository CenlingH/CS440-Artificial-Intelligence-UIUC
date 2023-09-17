from utils import compute_mst_cost
from abc import ABC, abstractmethod
import copy
# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0...
from itertools import count
global_index = count()

# TODO(III): You should read through this abstract class
#           your search implementation must work with this API,
#           namely your search will need to call is_goal() and get_neighbors()


class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f(state) = g(start, state) + h(state, goal)
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of AbstractState objects
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    # A* requires we compute a heuristic from each state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable, meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # The "hash" method allow us to keep track of which states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality

    @abstractmethod
    def __eq__(self, other):
        pass

# Grid ------------------------------------------------------------------------------------------------


class SingleGoalGridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of a single length 2 tuple location in the grid that needs to be reached, i.e., ((x,y),) 只是创建元组，所以多了个逗号
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        '''
        self.maze_neighbors = maze_neighbors
        # 调用父类构造函数
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO(V): implement this method
    def get_neighbors(self):
        nbr_states = []
        # We provide you with a method for getting a list of neighbors of a state,
        # you need to instantiate them as GridState objects
        # *self.state 将 (x, y) 这个元组解包为两个独立的参数，分别是 x 和 y
        neighboring_grid_locs = self.maze_neighbors(*self.state)
        for item in neighboring_grid_locs:
            nbr_states.append(SingleGoalGridState(
                item, self.goal, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors))
        return nbr_states

    # TODO(V): implement this method, check if the current state is the goal state
    def is_goal(self):
        return self.state == self.goal[0]

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    # TODO(V): implement this method
    # Compute the manhattan distance between self.state and self.goal
    def compute_heuristic(self):
        total = 0
        total = abs(self.state[0]-self.goal[0][0]) + \
            abs(self.state[1]-self.goal[0][1])
        return total

    # TODO(V): implement this method... should be unchanged from before
    def __lt__(self, other):
        # ctrl cv
        if self.dist_from_start+self.compute_heuristic() < other.dist_from_start+other.compute_heuristic():
            return True
        elif self.dist_from_start+self.compute_heuristic() == other.dist_from_start+other.compute_heuristic():
            if self.tiebreak_idx < other.tiebreak_idx:
                return True

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goal=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goal=" + str(self.goal)


def manhattan(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


class GridState(AbstractState):

    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, mst_cache=None):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of length 2 tuples location in the grid that needs to be reached
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
        '''
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO(VI): implement this method
    def get_neighbors(self):
        nbr_states = []
        # We provide you with a method for getting a list of neighbors of a state,
        # You need to instantiate them as GridState objects
        neighboring_locs = self.maze_neighbors(*self.state)
        for item in neighboring_locs:
            if item in self.goal:
                new_goal = copy.deepcopy(self.goal)
                goal_new = tuple(i for i in new_goal if i != item)
                nbr_states.append(GridState(item, goal_new, self.dist_from_start + 1,
                                  self.use_heuristic, self.maze_neighbors, self.mst_cache))
            else:
                nbr_states.append(GridState(item, self.goal, self.dist_from_start + 1,
                                  self.use_heuristic, self.maze_neighbors, self.mst_cache))
        return nbr_states

    # TODO(VI): implement this method
    def is_goal(self):
        return len(self.goal) == 0

    # TODO(VI): implement these methods __hash__ AND __eq__
    def __hash__(self):
        return hash(self.state)+hash(self.goal)

    def __eq__(self, other):
        return hash(self.state)+hash(self.goal) == hash(other.state)+hash(other.goal)

    # TODO(VI): implement this method
    # Our heuristic is: manhattan(self.state, nearest_goal) + MST(self.goal)
    # If we've computed MST(self.goal) before we can simply query the cache, otherwise compute it and cache value ，             !!!cache是一个字典，索引值为hash（self.goal）
    # NOTE: if self.goal has only one goal then the MST value is simply zero,
    #       and so the heuristic reduces to manhattan(self.state, self.goal[0])
    # You should use compute_mst_cost(self.goal, manhattan) which we imported from utils.py
    def compute_heuristic(self):
        if len(self.goal) == 0:
            return 0
        nearest_goal = self.goal[0]
        for item in self.goal:
            if manhattan(self.state, item) < manhattan(self.state, nearest_goal):
                nearest_goal = item

        mst = 0
        if len(self.goal) == 1:
            return abs(self.state[0]-self.goal[0][0]) + abs(self.state[1]-self.goal[0][1])
        if hash(self.goal) in self.mst_cache:
            mst = self.mst_cache[hash(self.goal)]
        else:
            mst = compute_mst_cost(self.goal, manhattan)
            self.mst_cache[hash(self.goal)] = mst
        return manhattan(self.state, nearest_goal) + mst
    # TODO(VI): implement this method... should be unchanged from before

    def __lt__(self, other):
        if self.dist_from_start+self.compute_heuristic() < other.dist_from_start+other.compute_heuristic():
            return True
        elif self.dist_from_start+self.compute_heuristic() == other.dist_from_start+other.compute_heuristic():
            if self.tiebreak_idx < other.tiebreak_idx:
                return True

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)

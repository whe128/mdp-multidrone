import time
import numpy as np

from multi_drone import MultiDroneUnc

class StateNode:
    def __init__(self, state: np.ndarray, actions_want_try: np.ndarray, parent: "StateNode" = None, done: bool = False, expansion_reward: float = 0.0):
        self.parent: StateNode = parent
        self.from_action_idx: int = -1                      # the index of the action taken from the parent to reach this node
        self.done: bool = done                              # whether this node represents a terminal state
        self.expansion_reward: float = expansion_reward     # immediate reward during expansion
        self.state: np.ndarray = state                      # state
        self.N_s = 0                                        # N(s): number of times this state has been visited

        # ============================= unexpanded actions  ============================= #
        # all actions that have not been expanded, use np.darray to store them
        self.untried_actions: np.ndarray = actions_want_try
        # number of untried action, idx ([0, all_action_count - 1] is valid range to get untried action
        all_action_count = len(self.untried_actions)
        # in the beginning, all actions are not expanded, its count is the number of all action
        self.untried_action_count = all_action_count

        # ============================= edge-related  ============================= #
        # due to node needs to be fully expanded, so we initial the length of 1D array is all_action_count
        # same idx on actions, N_s_a_s, Q_s_a_s and children means the one edge to corresponding child
        self.actions: np.ndarray = np.empty(all_action_count, dtype=np.int32)       # expanded action
        self.N_s_a_s: np.ndarray = np.zeros(all_action_count, dtype=np.int32)       # N(s, a)
        self.Q_s_a_s: np.ndarray = np.zeros(all_action_count, dtype=np.float32)     # Q(s, a)
        self.children: np.ndarray = np.empty(all_action_count, dtype=object)        # children
        # in the beginning, this node has no expanded node
        self.expanded_count = 0

    def add_child(self, child_node: "StateNode", action:int):
        """
        add a new expanded child node to this state node
        """
        # the index to store this new child
        insert_idx = self.expanded_count

        # ---------------------- link children and action ---------------------- #
        # add child and action into each 1D array
        self.children[insert_idx] = child_node
        self.actions[insert_idx] = action

        # update the child node from_action_idx (use to find position in the parent node when trace back)
        child_node.from_action_idx = insert_idx

        # Link back: set this node as the child's parent
        child_node.parent = self

        # update the expansion count
        self.expanded_count += 1

    def random_remove_action(self) -> int:
        """
        Randomly choose an action from the untried actions
        :return: the randomly chosen action
        """
        # randomly choose an index
        untried_idx = np.random.randint(self.untried_action_count)
        choose_action = self.untried_actions[untried_idx]

        # do not rearrange the array, just put the last untried action
        # into the removed position, keep the front part valid
        last_idx = self.untried_action_count - 1
        self.untried_actions[untried_idx] = self.untried_actions[last_idx]

        # update the count of untried actions
        self.untried_action_count -= 1

        return choose_action

    def is_fully_expanded(self) -> bool:
        """
        check whether this node has been fully expanded, if it is fully expanded.
        It means there is no more untried actions left. untried action count will be 0.
        :return: True if this node has been fully expanded, False otherwise.
        """
        return self.untried_action_count == 0


    def is_leaf_node(self) -> bool:
        """
        check whether this node has a leaf node. If it has no children, it means there
        is no more untried actions left. expanded count will be 0.
        :return: True if this node has a leaf node, False otherwise.
        """
        return self.expanded_count == 0


class MCTTree:
    def __init__(self, root_state: np.ndarray,
                 exploration_coe: float,
                 max_simulation_step:int,
                 env: MultiDroneUnc):
        self.exploration_coe: float = exploration_coe           # exploration coefficient in MCTS
        self.max_simulation_step: int = max_simulation_step     # maximum number of steps for each simulation
        self.num_actions: int = env.num_actions
        self.drone_num: int = root_state.shape[0]               # Number of drones, obtained from state row num
        self.action_base: int = int(round(env.num_actions ** (1.0 / self.drone_num))) # env only give total action num
        self.step_cost: int = env.get_config().step_cost        # cost per step
        self.simulate = env.simulate                            # simulate function in env
        self.discount_factor: float = env.get_config().discount_factor      # discount factor for future reward (gamma)
        self.obstacles: list[list] = env.get_config().obstacle_cells        # list of obstacle positions
        self.grid_size: list = env.get_config().grid_size                   # grid size [x, y ,z]
        self.goal_positions: list[list] = env.get_config().goal_positions   # list of goal positions

        self.greedy_epsilon = 0.2               # coe for exploration
        self.max_branch = 20                    # max branch
        self.goal_weight = 2
        self.obstacle_weight = 1

        # Initialize the root node of the MCTS tree
        self.root = StateNode(root_state, self.generate_try_expansion_actions(root_state, self.max_branch))


    def selection(self) -> StateNode:
        """selection process of MCTS tree"""

        # start from the root node
        current_node = self.root

        # Traverse the tree until a leaf node or unfully expanded node is reached
        # if current node is leaf node, stop loop
        while not current_node.is_leaf_node():
            # in this loop, the current node has children (not leaf)
            # this state node is not fully expanded
            if not current_node.is_fully_expanded():
                return current_node

            # at here the state node is fully expanded
            # choose the children by ucb:np.ndarray
            ucb = (current_node.Q_s_a_s
                   + self.exploration_coe
                   * np.sqrt(np.log(current_node.N_s) / current_node.N_s_a_s))
            # find the idx of max ucb -> use idx to find children
            best_action_idx = np.argmax(ucb)

            #update the current node
            current_node = current_node.children[best_action_idx]

        # at here this is leave node
        return current_node

    def expansion(self, select_node: StateNode) -> StateNode:
        '''
        expansion state node from the select node. Randomly choose an action
        from untried actions, and expand the next state node.
        :param select_node: the state node selected by selection.
                            Assume it is not the done state node.
        :return: choose an action, run the simulator get a next state node.
        '''

        # choose an action randomly
        action = select_node.random_remove_action()

        # run simulate to get next state (expansion)
        expansion_state, reward, done, _ = self.simulate(select_node.state, action)

        if done:
            # done state do not need to expand its children
            actions_want_try = np.array([], dtype=np.int32)
        else:
            actions_want_try = self.generate_try_expansion_actions(expansion_state, self.max_branch)
        expansion_node = StateNode(expansion_state,
                                   actions_want_try,
                                   parent = select_node,
                                   done = done,
                                   expansion_reward = reward + (self.step_cost * self.drone_num))  # add step cost back, the reward for the node

        # add expansion state node to select node
        select_node.add_child(expansion_node, action)

        return expansion_node

    def simulation(self, expanded_node: StateNode) -> float:
        """
        Run simulation from the expansion node, get the simulated total reward.
        :param expanded_node: the state node expanded by expansion.
                                Assume it is not the done state node.
        :return: the total reward of simulation from the expansion state node
        """

        # record the current state at each step and run simulation from it
        current_state = expanded_node.state
        # record the done information at each simulation step
        current_done = expanded_node.done
        # record how many steps have been run
        step_count = 0
        # cumulate discounted reward at each simulation step
        total_reward = 0.0

        # Simulate until reaching a terminal state (done) or maximum steps

        while not current_done and step_count < self.max_simulation_step:
            # ============= Strategy choose action for simulation ============= #
            # 1: random choose an action
            # action = np.random.randint(self.env.num_actions)
            t = time.time()
            # 2: heuristic-based action -> probably better action to get high reward
            simulation_action = self.heuristic_simulation_action(current_state)
            t_heuristic = time.time() - t

            # execute simulation based on state and action

            next_state, reward, next_done, _ = self.simulate(current_state, simulation_action)
            t_simulate = time.time() - t - t_heuristic
            # accumulate discounted reward
            total_reward += (self.discount_factor ** step_count) * reward

            # update -> move to next state
            current_state = next_state
            current_done = next_done
            step_count += 1
            t_total = time.time() - t

            #(f"heu: {t_heuristic}, simu: {t_simulate}, total : {t_total}")
        return total_reward

    def back_propagation(self, start_node: StateNode, reward: float):
        """
        Run backpropagation from start node to root node.
        :param start_node: may come from selection node,
                            expansion node with simulation,
                            expansion node without simulation.
        :param reward: the reward of the current state.
        """

        # record the current node at each back propagation step, first from start node
        current_node = start_node

        # start from the start_node, increment its N(s) initially
        current_node.N_s += 1

        # reward for the start node
        current_reward = reward

        # loop until to the root node
        while current_node.parent is not None:
            # get parent node and action id-> can find child in parent's 1D array
            parent_node = current_node.parent
            action_idx = current_node.from_action_idx

            # increment N(s, a), use action idx to find the position
            n_s_a = parent_node.N_s_a_s[action_idx]
            n_s_a += 1
            parent_node.N_s_a_s[action_idx] = n_s_a

            # update Q(s, a) average total discounted reward
            # reduce the step cost for each step
            current_reward = current_reward - self.step_cost * self.drone_num

            # discount the reward for each step back to the parent node
            current_reward = current_reward * self.discount_factor

            # (Q_old * n_old + reward) /(n_old + 1)
            # = Q_old + (reward - Q_old) / (n_old + 1)
            # = Q_old + (reward - Q_old) / (n_new)
            q_old = parent_node.Q_s_a_s[action_idx]
            parent_node.Q_s_a_s[action_idx] = q_old + (current_reward - q_old) / n_s_a

            # update the N(s) for parent node
            parent_node.N_s += 1

            # back to the parent node, back propagation
            current_node = parent_node

    def generate_try_expansion_actions(self, current_state: np.ndarray, max_branch: int) -> np.ndarray:
        """
        generate actions under the number of max branch, use the heuristic top order actions of each drone, choose
        some of them and let (top k) the whole combined action num less than max branch.
        use epsilon coe for exploration and 1 - epsilon probability for heuristic choice with top k actions.
        """
        # generate valid actions that are sorted with heuristic score order ([0] max score)
        score_ordered_actions_each: list[list[int]] = self.heuristic_ordered_valid_actions(current_state)

        # create the tuple to record the drone id
        drone_actions_tuple: list[tuple[int, list[int]]] = \
            [(drone_idx,actions) for drone_idx,actions in enumerate(score_ordered_actions_each)]

        # record the strategy each action choose how many top_k actions (drone, top_k)
        k_a_record: list[tuple[int, int]] = []

        # record the actions for each done that has not set the number about how many number need to choose
        remain_actions_tuple: list[tuple[int, list[int]]] = drone_actions_tuple
        remain_branch = float(max_branch)
        top_k = 0

        # ----------------------------------- choose top k ------------------------------- #
        # keep loop if there are action tuple left
        while remain_actions_tuple:
            # average k
            top_k = int(remain_branch ** (1.0 / len(remain_actions_tuple)))

            # store which drone actions can be chosen all or partly
            less_equal_k_actions = []
            upper_k_actions = []

            for actions_tuple in remain_actions_tuple:
                # action num less than top k, choose all actions
                if len(actions_tuple[1]) <= top_k:
                    less_equal_k_actions.append(actions_tuple)
                else:
                    # choose part actions
                    upper_k_actions.append(actions_tuple)

            # no action number less than k top, stop separation
            if not less_equal_k_actions:
                break

            # how many action combinations for drones have been chosen
            fixed_product = 1
            for action_idx, actions_list in less_equal_k_actions:
                action_num = len(actions_list)
                k_a_record.append((action_idx, action_num))
                fixed_product *= action_num

            # calculate the remaining max branch
            remain_branch /= fixed_product
            remain_actions_tuple = upper_k_actions

        # ensure not empty
        if remain_actions_tuple:
            # in k_a_record now, all drones can choose all actions
            # now the remaining actions need to choose top_k+1 partially (maybe)
            top_k_1_num = 0
            a_remain_count = len(remain_actions_tuple)
            for num_k_1 in range(1, a_remain_count + 1):
                num_k = a_remain_count - num_k_1
                # some set top k +1 but cannot let whole combination action
                # num exceed max branch remain_branch
                if ((top_k + 1) ** num_k_1) * (top_k ** num_k) > remain_branch:
                    top_k_1_num = num_k_1 - 1
                    break

            # ----------------------- set top k value of remain drone actions
            # now we know the determined drone actions,
            # and how many drones choose top_k, how many choose top_k + 1
            # random choose keep_top_k_num
            top_k_1_indices = np.random.choice(len(remain_actions_tuple), size=top_k_1_num, replace=False)

            # use the random index to separate the top k of drones has not been decided
            for idx, action_tuple in enumerate(remain_actions_tuple):
                drone_idx, actions = action_tuple
                if idx in top_k_1_indices:
                    # set top k+ 1
                    k_a_record.append((drone_idx, top_k + 1))
                else:
                    # top k
                    k_a_record.append((drone_idx, top_k))


        # --------------------------------------- choose action with top k --------------------------------------- #
        # use the top k value to random choose the corresponding actions for each drone
        # combine exploitation (heuristic score top k good) and exploration (each action space random)
        choose_action_vector = []

        # use the tuple[0], it is drone_idx, sort it for order and easily encoding
        k_a_record.sort(key=lambda x: x[0])
        for drone_idx, k in k_a_record:
            actions = score_ordered_actions_each[drone_idx]
            # choose all actions
            if len(actions) == k:
                choose_action_vector.append(actions)
            else:
                # exploration, random choose left t
                if np.random.rand() < self.greedy_epsilon:
                    random_vector = np.random.choice(actions, size = k, replace = False)
                    choose_action_vector.append(random_vector.astype(int).tolist())
                else:
                    # choose the top k actions slice the list
                    choose_action_vector.append(actions[:k])

        # get the integer action encoding for multi-drones
        encode_actions:list = self.generate_encode_actions(choose_action_vector)

        # random select remain actions -> to reach the max branch
        encode_actions_set = set(encode_actions)

        # some drones may have less action number than top_k (needed to select)
        # whole action may not reach max branch, random select the rest actions
        if not all([len(score_ordered_actions_each[drone_idx]) == k
                    for drone_idx, k in k_a_record]):
            for _ in range(max_branch - len(encode_actions_set)):
                while True:
                    random_action = np.random.randint(self.num_actions)
                    if random_action not in encode_actions_set:
                        encode_actions_set.add(random_action)
                        break

        return np.array(list(encode_actions_set))


    def generate_encode_actions(self, action_spaces: list[list]) -> list:
        """
        encode the action vectors into integer actions in a list
        :param action_spaces: the action vector of all drones
        :return: the encoded actions in a list
        """
        # --------------------- generate available action through each action space--------------
        available_actions = []  # store available actions (encoded)
        # index tracker for each drone
        each_drone_action_idx = [0 for _ in range(self.drone_num)]
        # calculate each base [base^0, base^1, base^2 ...]
        bases = [self.action_base ** drone_idx for drone_idx in range(self.drone_num)]  # each drone encoding base

        # traverse all combination of the vector
        while True:
            action_code = 0
            for i in range(self.drone_num):
                # action_0 * base^0 + action_1 * base^1 ....
                action_code += action_spaces[i][each_drone_action_idx[i]] * bases[i]
            available_actions.append(action_code)

            # choose next action
            for i in range(self.drone_num):
                each_drone_action_idx[i] += 1
                # can choose next action for this drone
                # current drone still has next action, no carry on,
                # next drones will not increment action index, break
                if each_drone_action_idx[i] < len(action_spaces[i]):
                    break
                else:
                    # carry on, reset to 0, and increment next drone action index
                    each_drone_action_idx[i] = 0
                    # if the last drone create carry on, all combinations have been appended
                    if i == self.drone_num - 1:
                        return available_actions

    def heuristic_simulation_action(self, current_state: np.ndarray) -> int:
        '''
        choose the simulation action through best heuristic score or random
        :return: heuristic action
        '''
        # get ordered actions
        score_ordered_actions_each: list[list] = self.heuristic_ordered_valid_actions(current_state)

        # heuristic_action_multi is encoded, here choose each
        # drone's action and encode into encoded action
        heuristic_action_multi = 0

        # compute heuristic for each drone
        for drone_idx in range(self.drone_num):

            # ------------------- epsilon process ------------------- #
            # get the max heuristic score of action
            # with the int action -> 0 1 2 3 4 ...
            order_actions_single = score_ordered_actions_each[drone_idx]
            best_action_encode_single = order_actions_single[0]
            # use epsilon for random action choice, other use the action with the best heuristic score
            if np.random.rand() < self.greedy_epsilon:  # epsilon_random_rollout, random exploration
                # all other actions that are not best score action
                candidate_actions = order_actions_single[1:]
                if len(candidate_actions) > 0:
                    next_action = np.random.choice(candidate_actions)
                else:
                    next_action = best_action_encode_single
            else:
                next_action = best_action_encode_single

            # encode single drone's action into centralized drones action
            heuristic_action_multi += next_action * (self.action_base ** drone_idx)

        return heuristic_action_multi

    def heuristic_ordered_valid_actions(self, current_state: np.ndarray) -> list[list]:
        '''
        consider the distance with goal and obstacles after the action
        :return: heuristic action
        '''

        # create the action vectors for different action base
        if self.action_base == 26:
            # 26 directions
            action_vectors = np.array(
                [[dx, dy, dz]
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)],
                dtype=np.int32
            )
        else:
            # 8 directions
            action_vectors = np.array(
                [[dx, dy, 0]
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 if not (dx == 0 and dy == 0)],
                dtype=np.int32
            )

        obstacle_num = len(self.obstacles)

        # heuristic_action_multi is encoded, here choose each
        # drone's action and encode into encoded action
        heuristic_action_multi = 0

        ordered_score_valid_actions = []
        # compute heuristic for each drone
        for drone_idx in range(self.drone_num):

            # reached goal drone do not need to move, lock as 0.
            if current_state[drone_idx][3].astype(bool):
                ordered_score_valid_actions.append([0])
                continue

            position = current_state[drone_idx, :3]  # current [x, y, z] of drone i
            goal_position = self.goal_positions[drone_idx]  # goal position of drone i

            # use numpy to compute all possible next position after different action
            next_positions = position + action_vectors

            # get the invalid actions - store the encode action
            invalid_actions = []

            for idx, pos in enumerate(next_positions):
                x, y, z = pos
                if (x < 0 or x >= self.grid_size[0]
                        or y < 0 or y >= self.grid_size[1]
                        or z < 0 or z >= self.grid_size[2]):
                    # out of bound
                    invalid_actions.append(idx)
                    continue

                # traverse each obstacle and if collision with one obs
                for obstacle in self.obstacles:
                    obs_x, obs_y, obs_z = obstacle
                    if x == obs_x and y == obs_y and z == obs_z:
                        invalid_actions.append(idx)
                        break

            # compute min-step max(|x2 - x1|, |y2 - y1|, |z2 - z1|) distance from each next positon to goal position
            # since the drone can run diagonally, the distance is determined by the largest coordinate difference
            goal_distance = np.max(np.abs(next_positions - goal_position), axis=1)

            # compute Manhattan distance from each next position to each obstacle
            # ob1: [ [a1_dis, a2_dis, ... ]
            # ob2:   [a1_dis, a2_dis, ... ]
            # ob3:   [a1_dis, a2_dis, ... ] ]
            # ...
            obstacle_distances = np.empty((obstacle_num, len(action_vectors)), dtype=int)
            for obs_idx, obstacle in enumerate(self.obstacles):
                # compute manhattan distance for each obstacle
                obstacle_dis = np.sum(np.abs(next_positions - obstacle), axis=1)
                # put into the 2D array
                obstacle_distances[obs_idx] = obstacle_dis

            # choose the closet obstacle distance for each action, through each col
            min_obstacle_distance = np.min(obstacle_distances, axis=0)

            # ======================== Heuristic score ======================== #
            # goal distance: bigger score lower -> high score need to go toward goal
            # obstacle distance: bigger score higher ->high score need to go away obstacle
            # closer to obstacle -> the uncertainty higher-> risk improve
            heuristic_score = - self.goal_weight * goal_distance + self.obstacle_weight * min_obstacle_distance

            valid_action_scores = [(a, heuristic_score[a]) for a in range(len(heuristic_score)) if a not in invalid_actions]

            # sort the score list
            valid_action_scores.sort(key=lambda x: x[1], reverse=True)
            ordered_actions = [a for a, _ in valid_action_scores]

            ordered_score_valid_actions.append(ordered_actions)

        return ordered_score_valid_actions

    def choose_best_action(self)->int:
        """
        best action under the root node, choose the action with highest Q(s, a)
        :return:
        """
        # get the index of highest Q(s, a), root may not be fully expanded
        # just consider the range from [0, expanded_count-1]
        best_action_idx = np.argmax(self.root.Q_s_a_s[:self.root.expanded_count])
        #(f"{self.root.expanded_count}                  {self.root.Q_s_a_s[best_action_idx]}")
        return int(self.root.actions[best_action_idx])

    def root_contain_children(self, action: int, current_state: np.ndarray)->StateNode:
        """
        check if the root node has a child corresponding to the action
        actually get the given state.
        :param action: the action to check
        :param current_state: the state to match the state in the child node
        :return: the child node if contained
        """
        action_idx = None
        # find the index of action among expanded actions
        for i in range(self.root.expanded_count):
            if self.root.actions[i] == action:
                action_idx = i
                break

        # not found, not contain this type of child
        if action_idx is None:
            return None

        # use the action index to get the corresponding child node
        child_node = self.root.children[action_idx]
        # compare the two state (2D array)
        if np.array_equal(child_node.state, current_state):
            return child_node

        # not match the state
        return None

    def cut_tree_from_node(self, node):
        """
        set the given node as the new root node of the MCTS tree
        """
        self.root = node
        # disconnect the node with its parent, avoid the propagation to parent
        node.parent = None

class MyPlanner:
    def __init__(self, env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env             # environment of multi-drones
        self._a_param = a_param     # a_param for exploration
        self._b_param = b_param     # b_param for max simulation step

        # record the MCTS tree and last action, use for check whether the action
        # and curren state in the MCTS tree for last planning
        self.mcts_tree = None  # MCTS tree instance
        self.last_action = None

    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:

        # ======================== Tree Initialization / Reuse ======================== #
        if self.mcts_tree is None:
            # create new tree - not exist
            self.mcts_tree = MCTTree(current_state, self._a_param, self._b_param, self._env)
        else:
            # tree exists, check last action and current state in the tree
            contains_node = self.mcts_tree.root_contain_children(self.last_action, current_state)
            if contains_node is None:
                # not in the tree, cannot reuse, create a new tree from the current state
                self.mcts_tree = MCTTree(current_state, self._a_param, self._b_param, self._env)
            else:
                # reused the subtree
                self.mcts_tree.cut_tree_from_node(contains_node)


        # ======================== MCTS iteration steps======================== #
        start_time = time.time()
        # each planning has time limit
        while time.time() - start_time < planning_time_per_step:
            #---------------------- Selection step --------
            select_node = self.mcts_tree.selection()

            # meet done state, directly go backpropagation
            if select_node.done:
                self.mcts_tree.back_propagation(select_node, select_node.expansion_reward)
                continue

            # ---------------------- Expansion step --------
            expansion_node = self.mcts_tree.expansion(select_node)

            # the expanded node is done state node, directly go backpropagation
            if expansion_node.done:
                self.mcts_tree.back_propagation(expansion_node, expansion_node.expansion_reward)
                continue

            # ---------------------- simulation step --------
            reward = self.mcts_tree.simulation(expansion_node)

            # ---------------------- backpropagation step --------
            self.mcts_tree.back_propagation(expansion_node, reward)

        # ======================== plan select ======================== #
        action = self.mcts_tree.choose_best_action()
        # save the last action for tree reuse in the next planning
        self.last_action = action

        return action

if __name__ == '__main__':
    def run(env, planner, planning_time_per_step=1.0):
        # Set the simulator to the initial state
        current_state = env.reset()
        num_steps = 0
        total_discounted_reward = 0.0
        history = []

        while True:
            # Use MCTS to plan an action from the current state
            action = planner.plan(current_state, planning_time_per_step)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Accumulate discounted reward
            total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

            # Log trajectory
            history.append((current_state, action, reward, next_state, done, info))

            # Move forward
            current_state = next_state
            num_steps += 1

            if done or num_steps >= env.get_config().max_num_steps:
                break

        return total_discounted_reward, history

    for i in range(1):
        env = MultiDroneUnc("example_config.yaml")
        # Instantiate the planner

        planner = MyPlanner(env, a_param=400, b_param=12)
        # Run the planning loop
        total_discounted_reward, history = run(env, planner, planning_time_per_step=1)
        print(f"success: {history[-1][5]['success']}, Total discounted reward: {total_discounted_reward}")
    env.show()

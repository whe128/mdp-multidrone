import time
import numpy as np
from multi_drone import MultiDroneUnc
from my_planner import MyPlanner

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
def mean_confidence_interval_95(data : list):
    if len(data) == 0:
        return 0, 0
    data_array = np.array(data)
    mean = np.mean(data_array)
    se = np.std(data_array) / np.sqrt(len(data_array))
    ci95 = 1.96 * se
    return mean, ci95

ENV_NAME = "config_q4_3.yaml"
TEST_ROUND = 30
UCB_COE_SINGLE_DRONE = 400
MAX_SIMULATION_STEPS = 12
PLANNING_TIME_PER_STEP = [0.1, 0.5, 1.0, 2.0, 5.0]

for planning_time_per_step in PLANNING_TIME_PER_STEP:
    env_success = []
    env_collision = []
    env_runtime = []

    env_total_steps = []
    env_success_steps = []

    env_total_discounted_reward = []
    env_success_discounted_reward = []


    for r in range(TEST_ROUND):
        env = MultiDroneUnc(ENV_NAME)
        print(f"\r[Q5_test] with [planning time]: {planning_time_per_step}      --test round {r+1}", end=" ")
        planner = MyPlanner(env, a_param=UCB_COE_SINGLE_DRONE, b_param=MAX_SIMULATION_STEPS)

        t_start = time.time()
        total_discounted_reward, history = run(env, planner, planning_time_per_step=planning_time_per_step)
        t_end = time.time()

        env_runtime.append(t_end - t_start)
        env_total_steps.append(len(history))
        env_total_discounted_reward.append(total_discounted_reward)


        if history[-1][5]['num_collisions'] + history[-1][5]['num_vehicle_collisions'] > 0:
            env_collision.append(1)
        else:
            env_collision.append(0)

        if history[-1][5]['success']:
            env_success.append(1)
            env_success_discounted_reward.append(total_discounted_reward)
            env_success_steps.append(len(history))
        else:
            env_success.append(0)

    # calculate statistic value
    mean_success, ci95_success = mean_confidence_interval_95(env_success)
    mean_collision, ci95_collision = mean_confidence_interval_95(env_collision)
    mean_time, ci95_time = mean_confidence_interval_95(env_runtime)

    mean_steps, ci95_steps = mean_confidence_interval_95(env_total_steps)
    mean_success_steps, ci95_success_steps = mean_confidence_interval_95(env_success_steps)

    mean_reward, ci95_reward = mean_confidence_interval_95(env_total_discounted_reward)
    mean_success_reward, ci95_success_reward = mean_confidence_interval_95(env_success_discounted_reward)





    print(f"\r[Q5 test with [planning time]: {planning_time_per_step}                       ")
    print(f" Success   rate: {mean_success * 100:.1f}% ± {ci95_success * 100:.1f}%")
    print(f" Collision rate: {mean_collision * 100:.1f}% ± {ci95_collision * 100:.1f}%")

    print(f" Runtime: {mean_time:.2f} ± {ci95_time:.2f} sec")
    print(f" Total   Steps: {mean_steps:.1f} ± {ci95_steps:.1f}")
    print(f" Success Steps: {mean_success_steps:.1f} ± {ci95_success_steps:.1f}")

    print(f" Total   Reward: {mean_reward:.1f} ± {ci95_reward:.1f}")
    print(f" Success Reward: {mean_success_reward:.1f} ± {ci95_success_reward:.1f}\n")


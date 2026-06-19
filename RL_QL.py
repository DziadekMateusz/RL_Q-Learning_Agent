import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Hyperparameters
GRID_SIZE = 10
EPISODES = 50000
SHOW_EVERY = 1000
MAX_STEPS = 200

DISCOUNT = 0.95
EPSILON_START = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.99995

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
OFFSET = GRID_SIZE - 1

q_table = np.full((2 * GRID_SIZE - 1,) * 4 + (4,), 5.0)


# Environment helpers
def random_pos():
    return np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)


def random_distinct_positions(n):
    "n distinct random grid cells (used for agent/goal/enemy)."
    positions = []
    while len(positions) < n:
        pos = random_pos()
        if pos not in positions:
            positions.append(pos)
    return positions


def step(pos, action):
    x, y = pos
    dx, dy = action
    return np.clip(x + dx, 0, GRID_SIZE - 1), np.clip(y + dy, 0, GRID_SIZE - 1)


def get_state(agent, goal, enemy):
    return (
        goal[0] - agent[0] + OFFSET,
        goal[1] - agent[1] + OFFSET,
        enemy[0] - agent[0] + OFFSET,
        enemy[1] - agent[1] + OFFSET,
    )


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Agent helpers
def best_action(q_values):
    max_q = np.max(q_values)
    candidates = np.where(q_values == max_q)[0]
    return np.random.choice(candidates)


def choose_action(state, epsilon):
    if random.random() > epsilon:
        return best_action(q_table[state])
    return np.random.randint(0, 4)


def compute_reward(agent, new_agent, goal, enemy, step_i):
    "Returns (reward, done, success)."
    if new_agent == goal:
        return 30, True, True
    if new_agent == enemy:
        return -30, True, False

    old_dg, new_dg = distance(agent, goal), distance(new_agent, goal)
    old_de, new_de = distance(agent, enemy), distance(new_agent, enemy)

    reward = -0.05
    reward += (old_dg - new_dg) * 0.6
    reward += (new_de - old_de) * 0.5
    reward -= step_i * 0.001
    if new_agent == agent:
        reward -= 0.7

    return reward, False, False


def update_q(state, action, reward, new_state, done, learning_rate):
    old_q = q_table[state + (action,)]
    max_future_q = 0 if done else np.max(q_table[new_state])
    new_q = (1 - learning_rate) * old_q + learning_rate * (reward + DISCOUNT * max_future_q)
    q_table[state + (action,)] = np.clip(new_q, -50, 50)


# Rendering
def render(agent, goal, enemy, scale=40):
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    img[goal[1]][goal[0]] = (0, 255, 0)
    img[enemy[1]][enemy[0]] = (0, 0, 255)
    img[agent[1]][agent[0]] = (255, 0, 0)
    img = cv2.resize(img, (GRID_SIZE * scale, GRID_SIZE * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("RL", img)
    cv2.waitKey(1)


# Training loop
results = []
success_counter = 0
last_result = 0
epsilon = EPSILON_START

for episode in range(EPISODES):
    learning_rate = max(0.05, 0.5 * (1 - episode / EPISODES))

    # adaptive epsilon (bump up if stagnating)
    if episode > 0 and episode % SHOW_EVERY == 0 and last_result < 200:
        epsilon = min(0.3, epsilon + 0.1)

    # decay
    epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)

    agent, goal, enemy = random_distinct_positions(3)
    should_render = episode % 5000 == 0

    for step_i in range(MAX_STEPS):
        state = get_state(agent, goal, enemy)
        action = choose_action(state, epsilon)
        new_agent = step(agent, ACTIONS[action])

        reward, done, success = compute_reward(agent, new_agent, goal, enemy, step_i)
        new_state = get_state(new_agent, goal, enemy)

        update_q(state, action, reward, new_state, done, learning_rate)

        agent = new_agent
        if should_render:
            render(agent, goal, enemy)

        if done:
            success_counter += success  # True == 1, False == 0
            break

    if (episode + 1) % SHOW_EVERY == 0:
        results.append(success_counter)
        print(f"{episode + 1}th episode: Success={success_counter}")
        last_result = success_counter
        success_counter = 0

cv2.destroyAllWindows()

plt.plot(results)
plt.xlabel("Episode blocks (0-50)")
plt.ylabel("Success rate (0-1000)")
plt.title("Q-learning accuracy")
plt.grid()
plt.show()
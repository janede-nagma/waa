import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Maze Environment
class MazeEnv:
    def __init__(self):
        # 0 = path, 1 = wall, 2 = goal
        self.maze = np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 2]
        ])
        self.start = (0, 0)
        self.goal = (4, 4)
        self.state = self.start
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        row, col = self.state
        
        # Move based on action
        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(4, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(4, col + 1)
        
        # Check if hit wall
        if self.maze[row, col] == 1:
            reward = -1
            row, col = self.state  # Stay in place
        elif self.maze[row, col] == 2:
            reward = 10  # Goal reached
        else:
            reward = -0.1  # Small penalty for each step
        
        self.state = (row, col)
        done = (self.state == self.goal)
        
        return self.state, reward, done

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = np.zeros((5, 5, 4))  # 5x5 maze, 4 actions
    
    def choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(4)  # Explore
        else:
            row, col = state
            return np.argmax(self.q_table[row, col])  # Exploit
    
    def train(self, episodes=1000):
        rewards_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 50:  # Max steps per episode
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Q-Learning update
                row, col = state
                next_row, next_col = next_state
                
                old_q = self.q_table[row, col, action]
                next_max_q = np.max(self.q_table[next_row, next_col])
                new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
                self.q_table[row, col, action] = new_q
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            rewards_history.append(total_reward)
            
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}")
        
        return rewards_history
    
    def get_policy(self):
        policy = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                if self.env.maze[i, j] != 1:
                    policy[i, j] = np.argmax(self.q_table[i, j])
        return policy

# Visualization
def visualize_maze(env):
    plt.figure(figsize=(6, 6))
    sns.heatmap(env.maze, annot=True, cmap="YlGnBu", cbar=False, 
                square=True, linewidths=1, linecolor='black')
    plt.title("Maze Environment (0=Path, 1=Wall, 2=Goal)")
    plt.savefig("maze_environment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Maze visualization saved")

def visualize_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(pd.Series(rewards).rolling(100).mean(), linewidth=2, label='100-episode average')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("learning_progress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Learning progress saved")

def visualize_policy(env, policy):
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw maze
    for i in range(5):
        for j in range(5):
            if env.maze[i, j] == 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
            elif env.maze[i, j] == 2:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='gold'))
            else:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lightblue'))
                ax.text(j + 0.5, i + 0.5, arrows[policy[i, j]], 
                       ha='center', va='center', fontsize=20)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Learned Policy (Black=Wall, Gold=Goal)")
    plt.savefig("learned_policy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Policy visualization saved")

# Main execution
if __name__ == "__main__":
    import pandas as pd
    
    print("=== Reinforcement Learning: Maze Environment ===\n")
    
    # Create environment and agent
    env = MazeEnv()
    agent = QLearningAgent(env, learning_rate=0.1, discount=0.9, epsilon=0.1)
    
    # Visualize maze
    visualize_maze(env)
    
    # Train agent
    print("\n🤖 Training Q-Learning Agent...\n")
    rewards = agent.train(episodes=1000)
    
    # Get learned policy
    policy = agent.get_policy()
    
    # Visualize results
    visualize_rewards(rewards)
    visualize_policy(env, policy)
    
    # Test the agent
    print("\n🎮 Testing Learned Policy:")
    state = env.reset()
    path = [state]
    steps = 0
    
    while steps < 20:
        action = np.argmax(agent.q_table[state[0], state[1]])
        state, reward, done = env.step(action)
        path.append(state)
        steps += 1
        
        if done:
            print(f"✅ Goal reached in {steps} steps!")
            print(f"Path taken: {path}")
            break
    
    if not done:
        print("❌ Failed to reach goal")
    
    print("\n🎉 Reinforcement Learning Complete!")
    print("📊 Visualizations saved: maze_environment.png, learning_progress.png, learned_policy.png")
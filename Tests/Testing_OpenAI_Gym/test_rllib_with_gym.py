
import gym
from ray.rllib.agents.ppo import PPOTrainer


# Define your problem using python and openAI's gym API:
class SimpleCorridor(gym.Env):
    """Corridor in which an agent must learn to move right to reach the exit.

    ---------------------
    | S | 1 | 2 | 3 | G |   S=start; G=goal; corridor_length=5
    ---------------------

    Possible actions to chose from are: 0=left; 1=right
    Observations are floats indicating the current field index, e.g. 0.0 for
    starting position, 1.0 for the field next to the starting position, etc..
    Rewards are -0.1 for all steps, except when reaching the goal (+1.0).
    """

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # left and right
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))

    def reset(self):
        """Resets the episode and returns the initial observation of the new one.
        """
        self.cur_pos = 0
        # Return initial observation.
        return [self.cur_pos]

    def step(self, action):
        """Takes a single step in the episode given `action`

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        # Walk left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Walk right.
        elif action == 1:
            self.cur_pos += 1
        # Set `done` flag when end of corridor (goal) reached.
        done = self.cur_pos >= self.end_pos
        # +1 when goal reached, otherwise -1.
        reward = 1.0 if done else -0.1
        return [self.cur_pos], reward, done, {}


# Create an RLlib Trainer instance.
trainer = PPOTrainer(
    config={
        # Env class to use (here: our gym.Env sub-class from above).
        "env": SimpleCorridor,
        # Config dict to be passed to our custom env's constructor.
        "env_config": {
            # Use corridor with 20 fields (including S and G).
            "corridor_length": 20
        },
        # Parallelize environment rollouts.
        "num_workers": 3,
        "framework": 'torch'
    })

# Train for n iterations and report results (mean episode rewards).
# Since we have to move at least 19 times in the env to reach the goal and
# each move gives us -0.1 reward (except the last move at the end: +1.0),
# we can expect to reach an optimal episode reward of -0.1*18 + 1.0 = -0.8
for i in range(5):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")



# =============================

from ray.rllib.agents.ppo import PPOTrainer

trainer = PPOTrainer(
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "Taxi_v3",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
    })


# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.

for i in range(5):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()
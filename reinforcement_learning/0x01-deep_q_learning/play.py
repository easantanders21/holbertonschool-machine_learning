#!/usr/bin/env python3
"""
display a game played by the agent trained by train.py
"""

import gym
from keras import optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

AtariProcessor = __import__('train').AtariProcessor
build_model = __import__('train').build_model
build_agent = __import__('train').build_agent


def playing():
    """
    display a game played by the agent trained by train.py
    """
    env = gym.make('ALE/Breakout-v5')
    env.reset()
    actions = env.action_space.n
    model = build_model(actions)
    dqn = build_agent(model, actions)
    dqn.compile(optimizers.Adam(lr=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    playing()

#!/usr/bin/env python3
"""
Module contains functions that build and train a DQNAgent that can play Atari's Breakout
"""
import gym
import numpy as np
from keras import Model
from keras.layers import Dense, Flatten, Conv2D, Input, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
# Frames
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """
    Defines Atari environment
    """
    def observation(self, img):
        """
        Preprocesses images by resizing and making into greyscale
        Args:
            img: nd.array
        Returns: preprocessed image
        """
        # Grab image from array
        img = Image.fromarray(img[0])
        # Resize
        img = img.resize((84, 84), Image.LANCZOS).convert('L')
        # Turn into array again
        preprocessed_img = np.array(img)
        preprocessed_img = preprocessed_img.astype('unit8')
        return preprocessed_img

    def process_state_batch(self, batch):
        """
        Converts to float32
        Args:
            batch: batch of images
        Returns: processed_batch
        """
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

    def process_reward(self, reward):
        """
        Processes reward between -1 and 1
        Args:
            reward: reward
        Returns: reward
        """
        return np.clip(reward, -1., 1.)


def build_model(actions, input_shape=(84, 84)):
    """
    Function that builds a CNN model as defined by the DeepMind resource
    Args:
        actions: The number of actions that can be taken
        input_shape: shape of image
    Returns: CNN model
    """
    comp_imput_shape = (WINDOW_LENGTH, ) + input_shape
    inputs = Input(comp_imput_shape)
    permute_layer = Permute((2, 3, 1,))(inputs)
    conv1 = Conv2D(32, kernel_size=8, strides=4, activation='relu',
                   data_format='channels_last')(permute_layer)
    conv2 = Conv2D(64, kernel_size=4, strides=2, activation='relu',
                   data_format='channels_last')(conv1)
    conv3 = Conv2D(64, kernel_size=3, strides=1, activation='relu',
                   data_format='channels_last')(conv2)
    flatten = Flatten()(conv3)
    dense = Dense(512, activation='relu')(flatten)
    outputs = Dense(units=actions, activation='linear')(dense)
    model = Model(inputs, outputs)
    return model


def build_agent(model, actions):
    """
    Function that builds a DQNAgent.
    Args:
        model: CNN model
        actions: The number of actions possible
    Returns: dqn
    """
    processor = AtariProcessor()
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=0.05, nb_steps=100000)

    dqn = DQNAgent(model=model, policy=policy, memory=memory,
                   processor=processor, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=100000,
                   gamma=0.99, target_model_update=1e-2,
                   train_interval=4, delta_clip=1.)
    return dqn


def train():
    """
    Function trains a DQNAgent to play Atari Breakout.
    """
    # Setup envirionment
    env = gym.make("Breakout-v0")

    # Set up model
    actions = env.action_space.n
    model = build_model(actions)
    model.summary()

    # Set up agent
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])

    # Train
    dqn.fit(env, nb_steps=4000000, log_interval=25000, visualize=False, verbose=2)

    # Save
    try:
        dqn.save_weights('policy.h5')
        print('Saved')
    except Exception as e:
        print('Not Saved')
        print(e)

    env.close()


if __name__ == "__main__":
    train()

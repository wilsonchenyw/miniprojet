"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym

from tamer.agent import Tamer
import tensorflow as tf
from tensorflow.keras.models import load_model

async def main():
    env = gym.make('MountainCar-v0')

    # hyperparameters
    discount_factor = 0.9
    epsilon = 0.8  # vanilla Q learning actually works well with no random exploration
    min_eps = 0.02
    num_episodes = 30
    tame = False  # set to false for vanilla Q `learning
    #is_expert = True
    mode = 3 #mode0: clavier, mode1: gest, mode2: parole, mode3: instruction
    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.3
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)
    #model = "./tamer/saved_models/MountainCar-v0_local_model_1543395776.18.h5"
    #model_expert = tf.keras.models.load_model(model)
    await agent.train(mode, model_file_to_save='autosave')#,is_expert=is_expert,model_expert=model_expert
    print('finish')
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=20)


if __name__ == '__main__':
    asyncio.run(main())





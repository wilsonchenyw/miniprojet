import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter
import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp,mode,*cap):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        if len(cap)>0 and mode == 1:
            cap = cap [0]
        if len(cap)>0 and mode == 2:
            mic = cap [0]
        if mode == 1 :
                mpHands = mp.solutions.hands  #methode utilisee
                hands = mpHands.Hands(static_image_mode=False, #parametres
                        max_num_hands=2, 
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)  
                # methode pour connecter les key points
                mpDraw = mp.solutions.drawing_utils    
        if mode ==2 :
            #google speech Recognizer
            r = sr.Recognizer()
        rng = np.random.default_rng()
        tot_reward = 0
        state = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                print(f' {ts}', end='')
                if self.tame:
                    self.env.render()

                # Determine next action
                action = self.act(state)
                #print('action',MOUNTAINCAR_ACTION_MAP[action])
                if self.tame:
                    disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)
                if not self.tame:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        
                        frame = None

                        time.sleep(0.01)  # save the CPU
                        if mode ==0:
                            human_reward = disp.get_scalar_feedback()
                        elif mode ==1:
                            human_reward = disp.get_gest_feedback(cap,mpHands,hands,mpDraw)
                        elif mode == 2:
                            human_reward = disp.get_parole_feedback(r,mic)
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            #print('humain reward',human_reward)
                            dict_writer.writerow(
                                {
                                    'Episode': episode_index + 1,
                                    'Ep start ts': ep_start_time,
                                    'Feedback ts': feedback_ts,
                                    'Human Reward': human_reward,
                                    'Environment Reward': reward
                                }
                            )
                            self.H.update(state, action, human_reward)
                            break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
        if not self.tame:
            if episode_index %200 == 0:
                self.play(n_episodes=1, render=True)

                return self.evaluate(n_episodes=30)
            else :
                return -200
        else:
            self.play(n_episodes=1, render=True)

            return self.evaluate(n_episodes=30)

    async def train(self, mode,model_file_to_save=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        # render first so that pygame display shows up on top
        reward_log = []
        self.env.render()
        disp = None
        if self.tame:
            # only init pygame display if we're actually training tamer
            from tamer.interface import Interface
            disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)
        
        if mode ==1:
            cap = cv2.VideoCapture(0) # use the default camera as the audio source
        if mode ==2:
            cap = sr.Microphone() # use the default microphone as the audio source
        if mode == 0:
            for i in range(self.num_episodes):
                reward_temps = self._train_episode(i, disp,mode)
                if i%200 == 0:
                    reward_log.append(reward_temps)
        else:
            for i in range(self.num_episodes):
                reward_log.append(self._train_episode(i, disp,mode,cap))

        print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)
        print(reward_log)

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                tot_reward += reward
                if render:
                    self.env.render()
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model

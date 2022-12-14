from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import gym
scaler = preprocessing.StandardScaler()

featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
def featurize_state(state):
        """ Returns the featurized representation for a state. """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
env = gym.make('MountainCar-v0')
state = env.reset()
#state = ((np.array([-0.50720584,  0.        ], dtype=np.float32), {}))
scaler.fit(state)
print(featurize_state(state))
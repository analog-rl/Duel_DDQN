import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense, Activation, Input, Lambda, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import numpy as np

import os
print ("$PATH = ", os.environ['PATH'])

# import os
# # os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32"
# os.environ["KERAS_BACKEND"] = "tensorflow"


def create_model():
    input_dim_orig = env.observation_space.shape

    S = Input(shape=input_dim_orig)
    h = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', activation='relu', init='normal')(S)
    h = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', activation='relu', init='normal')(h)
    h = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu', init='normal')(h)
    h = Flatten()(h)
    #TODO DDQN split here--
    # Value = Dense 512 -> Dense 1
    # Advantage = Dense 512 -> Dense n.actions
    h = Dense(512, activation='relu', init='normal')(h)
    V = Dense(env.action_space.n, activation='linear', init='normal')(h)
    model = Model(S, V)
    return model

#form paper
# def create_model():
#     # # input_dim_orig = [2] + list(env.observation_space.shape)
#     # input_dim_orig = [1] + list(env.observation_space.shape)
#     # input_dim = np.product(input_dim_orig)
#     # S = Input(shape=[input_dim])
#     # h = Reshape(input_dim_orig)(S)
#     input_dim_orig = env.observation_space.shape
#
#     S = Input(shape=input_dim_orig)
#     h = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', activation='relu')(S)
#     h = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')(h)
#     h = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu')(h)
#     h = Flatten()(h)
#     #TODO DDQN split here--
#     # Value = Dense 512 -> Dense 1
#     # Advantage = Dense 512 -> Dense n.actions
#     h = Dense(512, activation='relu')(h)
#     V = Dense(env.action_space.n, activation='linear', init='zero')(h)
#     model = Model(S, V)
#     return model

# note: i could not get this to work
def create_kerlym_model():
    # S = Input(shape=[env.observation_space.shape])
    # input_dim_orig = [2] + list(env.observation_space.shape)
    input_dim_orig = [1] + list(env.observation_space.shape)
    # input_dim_orig = list(env.observation_space.shape)
    input_dim = np.product(input_dim_orig)
    S = Input(shape=[input_dim])
    h = Reshape(input_dim_orig)(S)
    h = TimeDistributed( Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu'))(h)
    h = TimeDistributed( Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', activation='relu'))(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    V = Dense(env.action_space.n, activation='linear',init='zero')(h)
    model = Model(S, V)
    return model


# def create_model():
#     trainable = True
#     cnn_model = Sequential()
#     x = Input(shape=env.observation_space.shape)
#     if args.batch_norm:
#         h = BatchNormalization()(x)
#     else:
#         h = x
#     cnn_model.add(ZeroPadding2D(input_shape=env.observation_space.shape))
#     # cnn_model.add(h)
#     cnn_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), trainable=trainable, activation=args.activation))
#     cnn_model.add(Convolution2D(64, 4, 4, subsample=(2, 2), trainable=trainable, activation=args.activation))
#     cnn_model.add(Convolution2D(64, 3, 3, trainable=trainable, activation=args.activation))
#     # cnn_model.add(Flatten())  # 3136
#     # cnn_model.add(Dense(512, trainable=trainable))
#     # cnn_model.add(Activation(args.activation))
#     cnn_model.add(Flatten())
#
#     cnn_model.add(Dense(env.action_space.n + 1, activation='softmax'))
#
#     # init_norm = Gaussian(loc=0.0, scale=0.01)
#     # layers = []
#     # layers.append(Input(shape=env.observation_space.shape))
#     # # The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity.
#     # layers.append(Conv((8, 8, 32), strides=4, init=init_norm, activation=args.activation)(h), batch_norm=args.batch_norm)
#     # # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
#     # layers.append(Conv((4, 4, 64), strides=2, init=init_norm, activation=args.activation)(h), batch_norm=args.batch_norm)
#     # # This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier.
#     # layers.append(Conv((3, 3, 64), strides=1, init=init_norm, activation=args.activation)(h), batch_norm=args.batch_norm)
#     # # The final hidden layer is fully-connected and consists of 512 rectifier units.
#     # layers.append(Affine(nout=512, init=init_norm, activation=args.activation)(h), batch_norm=self.batch_norm)
#     # # The output layer is a fully-connected linear layer with a single output for each valid action.
#     # # layers.append(Affine(nout=num_actions, init = init_norm))
#
#     # y = Dense(env.action_space.n + 1)(h)
#     # if args.advantage == 'avg':
#     #     z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
#     #                output_shape=(env.action_space.n,))(y)
#     # elif args.advantage == 'max':
#     #     z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
#     #                output_shape=(env.action_space.n,))(y)
#     # elif args.advantage == 'naive':
#     #     z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
#     # else:
#     #     assert False
#     # cnn_model.add(z)
#     return cnn_model

# def old_createLayers():
#     x = Input(shape=env.observation_space.shape)
#     if args.batch_norm:
#         h = BatchNormalization()(x)
#     else:
#         h = x
#     for i in range(args.layers):
#         h = Dense(args.hidden_size, activation=args.activation)(h)
#         if args.batch_norm and i != args.layers - 1:
#             h = BatchNormalization()(h)
#     y = Dense(env.action_space.n + 1)(h)
#     if args.advantage == 'avg':
#         z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
#                    output_shape=(env.action_space.n,))(y)
#     elif args.advantage == 'max':
#         z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
#                    output_shape=(env.action_space.n,))(y)
#     elif args.advantage == 'naive':
#         z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
#     else:
#         assert False
#
#     return x, z


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_start_size', type=int, default=50000)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
# parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='rmsprop')
parser.add_argument('--ep_start', type=float, default=1)
parser.add_argument('--ep_end', type=float, default=0.1)
parser.add_argument('--ep_endt', type=float, default=1000000)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--target_net_update_frequency', type=int, default=5000) #10000
parser.add_argument('--replay_memory_size', type=int, default=50000) #1000000

parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment)
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Discrete)

if args.gym_record:
    env.monitor.start(args.gym_record, force=True)

model = create_model()
model.summary()
if (args.optimizer == "adam"):
    optimizer = Adam(lr=args.learning_rate)
elif (args.optimizer == "rmsprop"):
    optimizer = RMSprop(lr=args.learning_rate)
else:
    assert False, "invalid optimizer"
model.compile(optimizer=optimizer, loss='mse')

target_model = create_model()
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

total_reward = 0
timestep = 0
train_steps = 0

for i_episode in range(args.episodes):
    observation = env.reset()
    observation_t1 = observation
    episode_reward = 0

    # for t in range(args.max_timesteps):
    t = -1
    while True:
        t += 1
        if args.display:
            env.render()

        ep = args.ep_end + (args.ep_start - args.ep_end) * max(0, args.ep_endt - train_steps) / args.ep_endt

        if timestep < args.replay_start_size or np.random.random() < ep:
            action = env.action_space.sample()
            if args.verbose > 0:
                print "e:", i_episode, "e.t:", t, "action:", action, "random", " ep:", ep
        else:
            s = np.array([observation])
            q = model.predict(s, batch_size=1)
            action = np.argmax(q[0])
            if args.verbose > 0:
                print "e:", i_episode, "e.t:", t, "action:", action, "q:", q, " ep:", ep

        if len(prestates) >= args.replay_memory_size:
            delidx = np.random.randint(0, len(prestates) - 1 - args.batch_size)
            del prestates[delidx]
            del actions[delidx]
            del rewards[delidx]
            del poststates[delidx]
            del terminals[delidx]

        # prestates.append(observation.ravel())
        # prestates.append(observation.reshape([1] + list(env.observation_space.shape)))
        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if args.verbose > 1:
            print("reward:", reward)

        rewards.append(reward)
        # poststates.append(observation.ravel())
        # poststates.append(observation.reshape([1] + list(env.observation_space.shape)))
        poststates.append(observation)
        terminals.append(done)

        observation_t1 = observation

        timestep += 1

        if timestep > args.replay_start_size:
            train_steps += 1
            if timestep % args.update_frequency == 0:
                for k in xrange(args.train_repeat):
                    if len(prestates) > args.batch_size:
                        indexes = np.random.randint(len(prestates), size=args.batch_size)
                    else:
                        indexes = range(len(prestates))

                    pre_sample = np.array([prestates[i] for i in indexes])
                    post_sample = np.array([poststates[i] for i in indexes])

                    # a = pre_sample[0]
                    # qpre = model.predict(a)

                    qpre = model.predict(pre_sample)
                    qpost = target_model.predict(post_sample)
                    for i in xrange(len(indexes)):
                        if terminals[indexes[i]]:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
                        else:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
                    model.train_on_batch(pre_sample, qpre)

            if timestep % args.target_net_update_frequency == 0:
                if args.verbose > 0:
                    print('timestep:', timestep, 'DDQN: Updating weights')
                weights = model.get_weights()
                target_model.set_weights(weights)
                    # weights = model.get_weights()
                    # target_weights = target_model.get_weights()
                    # for i in xrange(len(weights)):
                    #     weights[i] *= args.tau
                    #     target_weights[i] *= (1 - args.tau)
                    #     target_weights[i] += weights[i]
                    # target_model.set_weights(target_weights)

        if done:
            break

    print("Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward))
    total_reward += episode_reward

print("Average reward per episode {}".format(total_reward / args.episodes))

if args.gym_record:
    env.monitor.close()

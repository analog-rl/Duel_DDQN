import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras import initializations
import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statbin import statbin


def createLayers():
    # def custom_init(shape, scale=0.01, name=None):
    #   return K.variable(np.random.normal(loc=0.0, scale=scale,  size=shape),
    #                     name=name)
    custom_init = lambda shape, name: initializations.normal(shape, scale=0.01, name=name)
    x = Input(shape=env.observation_space.shape)
    if args.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i in range(args.layers):
        h = Dense(args.hidden_size, activation=args.activation, init=custom_init)(h)
        if args.batch_norm and i != args.layers - 1:
            h = BatchNormalization()(h)
    y = Dense(env.action_space.n + 1)(h)
    if args.advantage == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
    else:
        assert False

    return x, z


def plot_stats(episode):
    # if self.ipy_clear:
    #     from IPython import display
    #     display.clear_output(wait=True)
    fig = plt.figure(1)
    fig.canvas.set_window_title("Training Stats for %s" % (episode))
    plt.clf()
    plt.subplot(2, 2, 1)
    stats["tr"].plot()
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc=2)
    plt.subplot(2, 2, 2)
    stats["ft"].plot()
    plt.title("Finishing Time per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Finishing Time")
    plt.legend(loc=2)
    plt.subplot(2, 2, 3)
    stats["maxvf"].plot2(fill_col='lightblue', label='Avg Max VF')
    stats["minvf"].plot2(fill_col='slategrey', label='Avg Min VF')
    plt.title("Value Function Outputs")
    plt.xlabel("Episode")
    plt.ylabel("Value Fn")
    plt.legend(loc=2)
    ax = plt.subplot(2, 2, 4)
    stats["cost"].plot2()
    plt.title("Training Loss")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    try:
        #                ax.set_yscale("log", nonposy='clip')
        plt.tight_layout()
    except:
        pass
    plt.show(block=False)
    plt.draw()
    plt.pause(0.001)

debug_scale = 1.0
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_start_size', type=int, default=50000 * debug_scale)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--max_timesteps', type=int, default=200)
# parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
# parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--activation', choices=['tanh', 'relu'], default='relu')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='rmsprop')
parser.add_argument('--ep_start', type=float, default=1)
parser.add_argument('--ep_end', type=float, default=0.1)
parser.add_argument('--ep_endt', type=float, default=1000000)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--update_frequency', type=int, default=4)
# parser.add_argument('--target_net_update_frequency', type=int, default=10000 * debug_scale)
parser.add_argument('--target_net_update_frequency', type=int, default=32)
parser.add_argument('--replay_memory_size', type=int, default=1000000 * debug_scale)
parser.add_argument('--plot_frequency', type=int, default=10)


parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment)
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Discrete)

if args.gym_record:
    env.monitor.start(args.gym_record, force=True)

x, z = createLayers()
model = Model(input=x, output=z)
model.summary()
optimizer = Adam(lr=args.learning_rate)
# optimizer = RMSprop(lr=args.learning_rate)
model.compile(optimizer=optimizer, loss='mse')

x, z = createLayers()
target_model = Model(input=x, output=z)
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

total_reward = 0
timestep = 0
train_steps = 0

stats = None
# if self.enable_plots:
if stats is None:
    stats = {
        "tr": statbin(args.plot_frequency),  # Total Reward
        "ft": statbin(args.plot_frequency),  # Finishing Time
        "minvf": statbin(args.plot_frequency),  # Min Value Fn
        "maxvf": statbin(args.plot_frequency),  # Min Value Fn
        "cost": statbin(args.plot_frequency),  # Loss
    }
for i_episode in range(args.episodes):
    observation = env.reset()
    episode_reward = 0
    episode_ave_max_q = 0
    episode_ave_min_q = 0
    episode_ave_cost = []
    ep_t = 0
    for t in range(args.max_timesteps):
        if args.display:
            env.render()

        ep = args.ep_end + (args.ep_start - args.ep_end) * max(0, args.ep_endt - train_steps) / args.ep_endt

        s = np.array([observation])
        q = model.predict(s, batch_size=1)
        if timestep < args.replay_start_size or np.random.random() < ep:
            action = env.action_space.sample()
            if args.verbose > 0:
                print "e:", i_episode, "e.t:", t, "action:", action, "random", " ep:", ep
        else:
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

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        if args.verbose > 1:
            print("reward:", reward)

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)

        timestep += 1

        episode_reward += reward
        episode_ave_max_q += np.max(q)
        episode_ave_min_q += np.min(q)
        ep_t += 1


        if timestep > args.replay_start_size:
            train_steps += 1
            if timestep % args.update_frequency == 0:
                for k in xrange(args.train_repeat):
                    if len(prestates) > args.batch_size:
                        # indexes = range(args.batch_size)
                        # indexes = np.random.choice(len(prestates), size=args.batch_size)
                        indexes = np.random.randint(len(prestates), size=args.batch_size)
                    else:
                        indexes = range(len(prestates))

                    pre_sample = np.array([prestates[i] for i in indexes])
                    post_sample = np.array([poststates[i] for i in indexes])
                    qpre = model.predict(pre_sample).copy()
                    qpost = target_model.predict(post_sample)
                    for i in xrange(len(indexes)):
                        if terminals[indexes[i]]:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
                        else:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
                    loss = model.train_on_batch(pre_sample, qpre)

                    # # Define cost and gradient update op
                    # a = tf.placeholder("float", [None, self.env[0].action_space.n])
                    # y = tf.placeholder("float", [None])
                    # action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
                    # cost = tf.reduce_mean(tf.square(y - action_q_values))
                    # optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    # grad_update = optimizer.minimize(cost, var_list=network_params)
                    # cost = ops["cost"].eval(session=self.parent.session, feed_dict=fd)
                    cost = loss
                    episode_ave_cost.append(cost)


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

    stats['tr'].add(episode_reward)
    stats['ft'].add(ep_t)
    stats['maxvf'].add(episode_ave_max_q / float(ep_t))
    stats['minvf'].add(episode_ave_min_q / float(ep_t))
    stats['cost'].add(np.mean(episode_ave_cost))
        # 'maxvf': episode_ave_max_q / float(ep_t),
        # 'minvf': episode_ave_min_q / float(ep_t),
        # 'cost': np.mean(episode_ave_cost)


    print("Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward))
    total_reward += episode_reward

    if (i_episode % args.plot_frequency == 0):
        plot_stats(i_episode)

print("Average reward per episode {}".format(total_reward / args.episodes))

if args.gym_record:
    env.monitor.close()



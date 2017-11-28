import numpy as npA
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import embed
from maze import *


n_hidden = 40
n_episodes = 5000
e0 = 0.2
gamma= 0.99
lr = 0.000001


width=6
height = 5

maze = Maze(height=height,
            width=width)

fig = plt.figure()
ax = fig.gca()
ax.axis('equal')
maze.draw_maze(ax)
fig.savefig('fig_maze')

# - `action`: 0 (up), 1 (down), 2 (right), 3 (left)


# Neural Network
pos = tf.placeholder(shape=[None,2,1,1],dtype=tf.float32)

# layer = tf.contrib.layers.fully_connected(inputs=pos,
#                                           num_outputs=30)
conv_layer = tf.contrib.layers.conv2d(inputs=pos,
                                      kernel_size=1,
                                      num_outputs=20,
                                      activation_fn=tf.sigmoid,
                                      weights_initializer=tf.random_normal_initializer)

conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
                                      kernel_size=2,
                                      num_outputs=20,
                                      activation_fn=tf.sigmoid,
                                      weights_initializer=tf.random_normal_initializer)
conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
                                      kernel_size=2,
                                      num_outputs=50,
                                      activation_fn=None,                                      
                                      weights_initializer=tf.random_normal_initializer)


# conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
#                                       kernel_size=40,
#                                       num_outputs=n_hidden)
layer = tf.contrib.layers.fully_connected(inputs=tf.reshape(conv_layer,shape=np.array([-1,2*50])),
                                          num_outputs=n_hidden)
                                          # weights_initializer=tf.random_normal_initializer)
layer = tf.contrib.layers.fully_connected(inputs=layer,num_outputs=n_hidden)
Qout = tf.contrib.layers.fully_connected(inputs=layer,
                                          num_outputs=4,
                                          activation_fn=None)
                                          # weights_initializer=tf.random_normal_initializer)

probs = tf.nn.softmax(Qout)

predict = tf.argmax(probs,axis=1)

nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
trainOp = trainer.minimize(loss)
init = tf.initialize_all_variables()



# sess = tf.Session()
# sess.run(init)

# action = [0]
# reward = 0
# maze.restart()
# xy = maze.state.copy().reshape(1,2)/10

# # a = xy.tolist()
# # a.append(action)
# # a = [item for sublist in a for item in sublist]
# # inputs = np.array(a).reshape(1,3)

# action,allQ = sess.run([predict,Qout],feed_dict={inputs: xy})
# action = 3
# # Perform step
# maze.take_action(action)

# # Get reward
# r,t = maze.get_positional_reward()

# # Add step penalty
# r += maze.step_penalty

# # Give reward for going right
# if action == 2:
#     reward += 0.0

# reward += r

# # Qnew_list = []
# # for possible_action in range(4):
# new_xy = maze.state.copy().reshape(1,2)/10
#     # new_a = xy.tolist()
#     # new_a.append(possible_action)
#     # new_a = [item for sublist in new_a for item in sublist]
#     # new_inputs = np.array(new_a).reshape(1,3)
#     # Qnew_list.append(sess.run(predict,feed_dict={pos: new_inputs}))

# # Get Q for new position
# Qnew = sess.run(Qout,feed_dict={inputs: new_xy})
# Qtarget = allQ.copy()
# Qtarget[0,action] = reward + gamma*np.max(Qnew)

# _,loss_value =sess.run([trainOp,loss],feed_dict={inputs: new_xy,nextQ:Qtarget})
# _,new_allQ = sess.run([predict,Qout],feed_dict={inputs: xy})

inspect_freq = 100

sess = tf.Session()
sess.run(init)

jlist = []
rlist = []

xy_memory = []
qtarget_memory = []
for i in range(n_episodes):
    maze.restart()


    if np.mod(i,100)==0:
        print('Training')
        xy_list = np.array(xy_memory).reshape((len(xy_memory),2)).reshape((len(xy_memory),2,1,1))
        Qtarget_list = np.array(qtarget_memory).reshape((len(qtarget_memory),4))
        m=0
        while m<50:
            _,loss_value =sess.run([trainOp,loss],feed_dict={pos: xy_list,nextQ:Qtarget_list})
            m += 1
    
        xy_memory = []
        qtarget_memory = []

    xy = maze.state.copy().reshape(1,2)# *[1/width,1/height]
    xy = np.array([xy[0][0],xy[0][1]]).reshape(1,2,1,1)

    reward = 0
    j = 0
    print('Episode %i/%i' %(i,n_episodes))
    while j<99:

        # Find next action for current state
        action,allQ,prob_dist,conv = sess.run([predict,Qout,probs,conv_layer],feed_dict={pos: xy})

        # e = (e0-1)/n_episodes * i+1

        # if np.mod(i,inspect_freq)!=0:
        action = np.random.choice(range(4),p=prob_dist[0])
        # #     # With probability e, take a random action
        if np.mod(i,inspect_freq)!=0:        
            if np.random.rand(1) < e0:
                # print('Random!')
                action = np.random.randint(4)


        # Perform step
        maze.take_action(action)

        # Get new position
        new_xy = maze.state.copy().reshape(1,2)# *[1/width,1/height]
        new_xy = np.array([new_xy[0][0],new_xy[0][1]]).reshape(1,2,1,1)

        # Get reward
        r,t = maze.get_positional_reward()

        # Add step penalty
        r += maze.step_penalty

        # Give reward for going right
        if action == 2:
            r += 0

        reward += r

        # Get Q for new position
        Qnew = sess.run(Qout,feed_dict={pos: new_xy})
        Qtarget = allQ.copy()

        if not t:
            Qtarget[0,action] = r + gamma*np.max(Qnew)
        else:
            Qtarget[0,action] = r

        xy_memory.append(xy)
        qtarget_memory.append(Qtarget)

        xy = new_xy.copy()

        if np.mod(i,inspect_freq)==0:
            print('reward = %4.4f' %(reward))
            print(xy)
            print(conv)
            print(allQ)
            print(prob_dist)
            maze.draw_maze(ax)
            plt.pause(0.001)
            input('Press enter to continue...')


        # If termination is True, break
        if t:
            jlist.append(j)
            rlist.append(reward)
            break


        j +=1


            
rlist = np.array(rlist)
jlist = np.array(jlist)

sess.close()


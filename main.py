import numpy as npA
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import embed
from maze import *


n_hidden = 200
n_episodes = 5000
e0 = 0.1
gamma= 0.99
lr = 0.1


width=10
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
pos = tf.placeholder(shape=[None,2],dtype=tf.float32)

# W = tf.Variable(tf.random_uniform([2,100],0,0.01))
# Qout = tf.matmul(inputs1,W)
layer = tf.contrib.layers.fully_connected(inputs=pos,
                                          num_outputs=n_hidden)
                                          # weights_initializer=tf.random_normal_initializer)
# layer = tf.contrib.layers.fully_connected(inputs=layer,num_outputs=n_hidden)
layer = tf.contrib.layers.fully_connected(inputs=layer,
                                          num_outputs=4,
                                          activation_fn=None)
                                          # weights_initializer=tf.random_normal_initializer)

Qout = tf.nn.softmax(layer)

predict = tf.argmax(Qout,axis=1)

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

with tf.Session() as sess:
    sess.run(init)
    
    jlist = []
    rlist = []    
    for i in range(n_episodes):
        maze.restart()

        xy_memory = []
        qtarget_memory = []

        xy = maze.state.copy().reshape(1,2)*[1/width,1/height]

        reward = 0
        j = 0
        print('Episode %i/%i' %(i,n_episodes))
        while j<99:

            # Find next action for current state
            action,allQ = sess.run([predict,Qout],feed_dict={pos: xy})




            # e = (e0-1)/n_episodes * i+1
            
            # if np.mod(i,500)!=0:
            action = np.random.choice(range(4),p=allQ[0])
            # #     # With probability e, take a random action
            #     if np.random.rand(1) < e:
            #         # print('Random!')
            #         action = np.random.randint(4)
            

            
            # Perform step
            maze.take_action(action)
            
            # Get new position
            new_xy = maze.state.copy().reshape(1,2)*[1/width,1/height]
            
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
                maze.draw_maze(ax)
                plt.pause(0.01)
                print(xy)
                embed()

            # If termination is True, break
            if t:
                jlist.append(j)
                rlist.append(reward)
                break

            
            j +=1

        xy_list = np.array(xy_memory).reshape((len(xy_memory),2))
        Qtarget_list = np.array(qtarget_memory).reshape((len(qtarget_memory),4))
        m=0
        while m<100:
            _,loss_value =sess.run([trainOp,loss],feed_dict={pos: xy_list,nextQ:Qtarget_list})
            m += 1            
            
rlist = np.array(rlist)
jlist = np.array(jlist)




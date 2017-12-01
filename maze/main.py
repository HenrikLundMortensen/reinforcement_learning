import numpy as npAAA
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import embed
from maze import *


n_hidden = 20
n_episodes = 10000
e0 = 0.1
gamma= 0.99
# lr = 0.000001


width=15
height = 4

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
                                      num_outputs=50,
                                      activation_fn=tf.sigmoid,
                                      weights_initializer=tf.random_normal_initializer)

# conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
#                                       kernel_size=2,
#                                       num_outputs=20,
#                                       activation_fn=tf.sigmoid,
#                                       weights_initializer=tf.random_normal_initializer)
conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
                                      kernel_size=2,
                                      num_outputs=5,
                                      activation_fn=None,                                      
                                      weights_initializer=tf.random_normal_initializer)


# conv_layer = tf.contrib.layers.conv2d(inputs=conv_layer,
#                                       kernel_size=40,
#                                       num_outputs=n_hidden)
layer = tf.contrib.layers.fully_connected(inputs=tf.reshape(conv_layer,shape=np.array([-1,2*5])),
                                          num_outputs=n_hidden)
                                          # weights_initializer=tf.random_normal_initializer)
layer = tf.contrib.layers.fully_connected(inputs=layer,
                                          num_outputs=n_hidden)
layer = tf.contrib.layers.fully_connected(inputs=layer,
                                          num_outputs=n_hidden)
# layer = tf.contrib.layers.fully_connected(inputs=layer,
#                                           num_outputs=n_hidden)
Qout = tf.contrib.layers.fully_connected(inputs=layer,
                                          num_outputs=4,
                                          activation_fn=None)
                                          # weights_initializer=tf.random_normal_initializer)
probs = tf.nn.softmax(Qout)

predict = tf.argmax(probs,axis=1)

nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ-Qout))
trainer = tf.train.AdamOptimizer()
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

inspect_freq = n_episodes-1

sess = tf.Session()
sess.run(init)

jlist = []
rlist = []
tmp_rlist= []
win = 0

xy_memory = []
qtarget_memory = []
for i in range(n_episodes):
    maze.restart()


    if np.mod(i,200)==0 and i!=0:
        # print('Training')
        xy_memory = np.array(xy_memory)
        qtarget_memory = np.array(qtarget_memory)
        
        xy_list = xy_memory.reshape(xy_memory.shape[0],xy_memory.shape[2],1,1)
        Qtarget_list = qtarget_memory.reshape(qtarget_memory.shape[0],qtarget_memory.shape[1]*qtarget_memory.shape[2])
        m=0
        while m<50:
            _,loss_value =sess.run([trainOp,loss],feed_dict={pos: xy_list,
                                                             nextQ:Qtarget_list})
            m += 1
    
        xy_memory = []
        qtarget_memory = []

    xy = maze.state.copy().reshape(1,2)
    # xy_one_hot = np.zeros(shape=(1,height+width))
    # xy_one_hot[0][xy[0][0]] = 1
    # xy_one_hot[0][xy[0][1]+width] = 1


    reward = 0
    j = 0
    if np.mod(i,20)==0:
        print('Episode %i/%i\t mean reward = %4.4f\t wins: %i' %(i,n_episodes,np.mean(tmp_rlist),win),end='\r')
        tmp_rlist = []
    while j<99:

        # Find next action for current state
        # action,allQ,prob_dist = sess.run([predict,Qout,probs],feed_dict={pos: xy})
        action,allQ,prob_dist = sess.run([predict,Qout,probs],feed_dict={pos: xy.reshape(1,2,1,1)})        



        # if np.mod(i,inspect_freq)!=0:
        action = np.random.choice(range(4),p=prob_dist[0])
        # #     # With probability e, take a random action

        # e = (-e0)/n_episodes * i+e0
        e = e0
        if np.mod(i,inspect_freq)!=0:        
            if np.random.rand(1) < e:
                # print('Random!')
                action = np.random.randint(4)


        # Perform step
        maze.take_action(action)

        # Get new position
        new_xy = maze.state.copy().reshape(1,2)# *[1/width,1/height]
        # new_xy_one_hot = np.zeros(shape=(1,height+width))
        # new_xy_one_hot[0][new_xy[0][0]] = 1
        # new_xy_one_hot[0][new_xy[0][1]+width] = 1




        # Get reward
        r,t = maze.get_positional_reward()

        # Add step penalty
        r += maze.step_penalty

        if r > 50:
            win +=1
        
        # Give reward for going right
        if action == 2:
            r += 0

        reward += r

        # Get Q for new position
        # Qnew = sess.run(Qout,feed_dict={pos: new_xy})
        Qnew = sess.run(Qout,feed_dict={pos: new_xy.reshape(1,2,1,1)})        
        Qtarget = allQ.copy()

        if not t:
            Qtarget[0,action] = r + gamma*np.max(Qnew)
        else:
            Qtarget[0,action] = r

        # _,loss_value =sess.run([trainOp,loss],feed_dict={pos: xy_one_hot.reshape(1,width*height),nextQ:Qtarget})
            
        xy_memory.append(xy)
        qtarget_memory.append(Qtarget)

        xy = new_xy.copy()
        xy_one_hot = np.zeros(shape=(1,height+width))
        xy_one_hot[0][xy[0][0]] = 1
        xy_one_hot[0][xy[0][1]+width] = 1

        if np.mod(i,inspect_freq)==0:
            print('reward = %4.4f' %(reward))
            print(xy)
            print(allQ)
            print(prob_dist)
            maze.draw_maze(ax)
            plt.pause(0.001)
            input('Press enter to continue...')



            
        # If termination is True, break
        if t:
            jlist.append(j)
            rlist.append(reward)
            tmp_rlist.append(reward)
            break


        j +=1


            
rlist = np.array(rlist)
jlist = np.array(jlist)

# sess.close()


import gym
import numpy as np
import numpy.random as npr
from scipy.linalg import expm


def decide_action(theta, state):
    s = state.reshape(dimension, 1)
    tmp = theta.T.dot(s)
    if(tmp < 0):
        return 0
    else:
        return 1
    
def set_weight(w, l):
    _sum = 0.0
    for i in range(l):
        _sum += np.maximum(0, np.log(l/2.0 + 1) - np.log(i + 1))
    for i in range(l):
        tmp = np.maximum(0, np.log(l/2.0 + 1) - np.log(i + 1))
        res = (tmp / _sum) - 1.0/l
        w.append(res)        
    return w

#変数

dimension = 4
num_of_sample = 8
M = 5
T = 200
alpha = 0.1
sigma = 0.5
B = np.matrix(np.identity(dimension))
max_time = 2000
max_trial = 1
success_count = 0
eta_m = 1.0
eta_b = (3.0*(3.0+np.log(dimension))) / (5.0 * dimension * np.sqrt(dimension))
log = ["gen,best_reward\n"]

for trial in range(max_trial):
    #npr.seed(trial)
    #初期化
    env = gym.make('CartPole-v0')
    theta = ((2 ** npr.rand(dimension)) - 10 * np.ones((dimension))).reshape(dimension,1)
    #theta = (10 * npr.rand(dimension) + np.ones(dimension)).reshape(dimension, 1)
    #theta = np.array([(npr.rand(dimension) - 0.5) * 0.1]).reshape(dimension, 1)
    
    children = []
    w = []
    set_weight(w, num_of_sample)
    
    for g in range(max_time):
        children.clear()
        
        for j in range(num_of_sample):
            z_j = npr.randn(dimension).reshape(dimension, 1)
            theta_j = theta + sigma * B.dot(z_j)
            children.append([z_j, theta_j, 0])
        for j in range(num_of_sample):
            c_j = children[j]
            for m in range(M):
                observation = env.reset()
                for t in range(T):
                    #env.render()
                    action = int(decide_action(c_j[1], observation))
                    observation, reward, done, info = env.step(action)
                    c_j[2] += reward
                    if(done):
                        break;
           #print("reward {}".format(c_j[2]))
        children = sorted(children, key=lambda x: x[2], reverse=True)
        log.append(str(g)+","+str(children[0][2])+"\n")
        if(children[0][2] >= T*M):
            break;
        #update
        G_m = np.zeros([dimension, 1])
        for i in range(num_of_sample):
            G_m += w[i] * children[i][0]
        G_m = np.dot(B, G_m)
        G_m *= sigma
        G_M = np.zeros([dimension, dimension])
        I = np.matrix(np.identity(dimension))
        for i in range(num_of_sample):
            tmp = children[i][0].dot(children[i][0].T)
            tmp2 = tmp - I
            tmp2 *= w[i]
            G_M += tmp2
        G_sigma = G_M.trace()/dimension
        G_B = G_M - G_sigma * I
        theta = theta + eta_m * G_m
        sigma = sigma * np.exp(G_sigma * eta_b / 2.0)
        B = B.dot(expm((eta_b/2.0) * G_B))
        if(g % 500 == 0):
            print(g)
    if(g == max_time - 1):
        print("fail")
    else:
        print("success {} gen".format(g+1))
        success_count += 1

print("finish all trial")
print("success:{} rate:{}".format(success_count, success_count/max_trial))
print("put log")
f = open('log.csv', 'w')
f.write(''.join(log))
f.close()

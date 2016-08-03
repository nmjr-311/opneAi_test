import gym
import numpy as np
import numpy.random as r

#var
dim = 4
M = 5
T = 200
alpha = 0.1
r_best = 0
max_gen = 2000
success_count = 0

#log var
renderFlag = False
logFlag = True

if logFlag:
    max_trial = 1
else:
    max_trial = 100
log = ["gen,best_reward\n"]

def decide_action(theta, state):
    s = state.reshape(dim, 1)
    tmp = theta.T.dot(s)
    if (tmp < 0):
        return 0
    else:
        return 1

#main loop
for tri in range(max_trial):
    #init
    env = gym.make('CartPole-v0')
    theta = (2*r.rand(dim) - 1 * np.ones(dim)).reshape(dim,1)
    r_best = 0
    #1 trial
    for g in range(max_gen):
        theta_new = theta + alpha * (2*r.rand(dim) - np.ones(dim)).reshape(dim, 1)
        r_total = 0
        for m in range(M):
            observation = env.reset()
            for t in range(T):
                if renderFlag:
                    env.render()
                action = int(decide_action(theta_new, observation))
                observation, reward, done, info = env.step(action)
                r_total += reward
                if(done):
                    break
        if r_total > r_best:
            r_best = r_total
            theta = theta_new
        if logFlag:
            log.append(str(g) + ',' + str(r_best) + '\n')
        if r_best >= T*M:
            break
    if(g == max_gen - 1):
        print('fail')
    else:
        print('success {} gen'.format(g+1))
        success_count += 1

print('finish all trial')
print('success:{} rate:{}'.format(success_count, success_count/max_trial))

if logFlag:
    print('put log')
    f = open('log_yama.csv', 'w')
    f.write(''.join(log))
    f.close()

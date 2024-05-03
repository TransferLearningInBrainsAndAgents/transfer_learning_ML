
import matplotlib.pyplot as plt
plt.rc('font', size=20)
import numpy as np
import copy

epsilon = 1e-3
# Set up the reward function
#C = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
C = np.array([[0, 1800], [1000, 0]])
taus = np.array([200, 50, 33, 25])
betas = 1/taus
#c = [copy.copy(C)]
c = [(C>1).astype(int)]
k = [(C>1).astype(float)]
# Setup the environment
T = 5000
#r_placements = np.array([[1, 10], [11, 20], [31, 40], [41, 50]])
r_placements = np.array([[1, 10], [12, 20]])

# Run the agent-environment interaction and collect the Reward (R) and the State (epsilon)
rs = np.zeros((C.shape[0], T))
rho = np.zeros((C.shape[0], T))
#trajectory = [np.random.randint(0, r_placements[-1][-1]+1, 1)[0]]
#trajectory = [5]
trajectory = 10 * np.sin(0.008 * np.arange(T)) + 11 #+ np.random.randint(-2, 2, T)
R = [0]

for t in np.arange(1, T):
    # Make a random move left or right or nothing
    '''
    new_pos = trajectory[-1] + np.random.randint(-1, 2, 1)[0]
    if new_pos < 0:
        new_pos = 0
    if new_pos > r_placements[-1][-1]+1:
        new_pos = r_placements[-1][-1]+1
    trajectory.append(new_pos)
    '''

    # Get the new r for each place
    for place_i in np.arange(len(r_placements)):
        if r_placements[place_i, 0] <= trajectory[t] <= r_placements[place_i, 1]:
            rs[place_i, t] = 1

    # Get the new c for each place combination and the new rho for each place
    c.append(copy.copy(c[t - 1]))
    k.append(copy.copy(k[t - 1]))
    R_temp = np.zeros(len(r_placements))
    for place_i in np.arange(len(r_placements)):
        for place_j in np.arange(len(r_placements)):
            c[t][place_i, place_j] = (np.max([rs[place_j, t], c[t-1][place_i, place_j]]) -
                                      np.min([rs[place_i, t], c[t-1][place_i, place_j]])) * (C[place_i, place_j] > 0) * \
                                    (1 - (1 if rho[place_i, t - 1] > epsilon else 0))

            delta_c = np.abs(c[t-1][place_i, place_j] - c[t][place_i, place_j])
            delta_k = (1 / np.max([C[place_i, place_j], 1]))
            k[t][place_i, place_j] = np.max([c[t][place_i, place_j] * (delta_c + k[t-1][place_i, place_j] - delta_k), 0])

        rho[place_i, t] = max([(1 - betas[place_i]) * (rho[place_i, t - 1] + rs[place_i, t] * np.sum(c[t-1][place_i, :])),
                               0])
        if rs[place_i, t] <= rho[place_i, t] / (1 - betas[place_i]) and \
                rho[place_i, t - 1] < epsilon and np.sum(k[t-1][place_i, :]) > 0:
            R_temp[place_i] = (rs[place_i, t])

    # Get the total given reward
    R.append(np.sum(R_temp))

rs = np.array(rs)
R = np.array(R)
c = np.array(c)
k = np.array(k)



plt.plot(R*30, c='k')
plt.plot(trajectory)
plt.plot(k[:, 0, 1]*20)
plt.plot(k[:, 1, 0]*20)
colours = ['r', 'b']
#colours = ['r', 'b', 'g', 'y']
for i, col in enumerate(colours):
    plt.plot(rho[i] * 20, c=col)
for i, rp in enumerate(r_placements):
    plt.fill_between(x=np.arange(T), y1=rp[0], y2=rp[1], alpha=0.4, color=colours[i])


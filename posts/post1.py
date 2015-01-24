import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import factorial
from scipy.optimize import curve_fit

import game
import driver


def poisson(x, *p):
    """Return poisson distribution."""
    A, lam, sigma = p
    return A * lam ** (x / sigma) * np.exp(-lam) / factorial(x / sigma)


metrics_list = ['bias']
metrics = game.Metrics2048(metrics_list)

i = 1
start_parameters = [0]
eta = 0.1
sim = driver.LearnGame(game.Game2048, metrics, eta=eta, start_parameters=start_parameters)
# score, board_metrics = sim.play_game()
sim.train(10000)

results = pd.DataFrame(sim.scores)
results['x1'] = np.array(sim.parameter_history)[:,0]
results.rename(columns={0: 'score', 'x1': 'x1'}, inplace=True)

results.score.hist(bins=100, normed=True)
plt.xlabel('Score')
plt.ylabel('Probability')
plt.title('Normalized Histogram of Scores, H = x1')
plt.show()

results.x1.plot()
plt.xlabel('Game')
plt.ylabel('x1')
plt.title('Value of x1 while training.')
plt.show()


# Mean x1 much less then mean and median score.
print('For games 9k-10k)')
print('Mean x1 {:.2f}'.format(results.x1.iloc[-1000:].mean()))
print('Mean score {:.2f}, median score {:.2f}'.
      format(results.score.iloc[-1000:].mean(), np.median(results.score.iloc[-1000:])))

# Distribution looks vaguely Poisson.
counts, bins = np.histogram(results.score, bins=100, normed=True)
bin_centers = (bins[1:] + bins[:-1])/2.0

# Rescale bin centers for fitting.
mean_score = np.mean(results.score)
x = bin_centers
coeff, var_matrix = curve_fit(poisson, x, counts, p0=[1.0, 1.0, mean_score])

y_fit = poisson(x, *coeff)

plt.plot(x, y_fit)
plt.bar(x, counts)
plt.xlabel('Score')
plt.ylabel('Probability')
plt.title('Value of x1 while training.')
plt.show()

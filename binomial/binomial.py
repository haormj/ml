from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    n, p = 5, 0.4
    x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
    ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
    plt.show()

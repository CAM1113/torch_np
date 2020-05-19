import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x + 10 * np.sin(x * 5) + 7 * np.cos(4 * x)

def main():
    x = np.arange(start=-10,stop = 10,step = 0.0001)
    y = func(x)
    # plt.scatter(x,y)
    plt.plot(x,y,)
    plt.show()

if __name__ == '__main__':
    main()

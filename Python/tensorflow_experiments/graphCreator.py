from matplotlib import pyplot as plt
import numpy as np

class Grapher():
    @staticmethod
    def create(name: str, interval: int, data: np.ndarray) -> None:
        new_data = np.zeros((np.size(data) // interval, 2))
        for i in range(np.size(data) // interval):
            new_data[i, :] = [i, np.mean(data[interval * i: interval * (i+1), 1])]

        figure, axis = plt.subplots(ncols=2)

        axis[0].plot(data[:, 0], data[:, 1])
        axis[0].set_xlabel("Episode")
        axis[0].set_ylabel("Cumulative reward")

        axis[1].plot(new_data[:, 0], new_data[:, 1])
        axis[1].set_xlabel(f"{interval}s of episodes")
        axis[1].set_ylabel("Average cumulative reward")

        plt.tight_layout()
        plt.savefig(f"Python/graphs/{name}.png", pad_inches=2)
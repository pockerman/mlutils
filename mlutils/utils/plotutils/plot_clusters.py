import matplotlib.pyplot as plt
from typing import List

def plot_clusters(data: List, data_colors: List, *, centers: List,
                  data_color_map: str='viridis',
                  data_s: int=200, centers_color: str='black',
                  centers_s: int=200,
                  centers_alpha:float=0.5,
                  plot_title: str=None,
                  plot_labels: List[str]=None) -> None:
    plt.scatter(data[:, 0], data[:, 1], c=data_colors, s=data_s, cmap=data_color_map)
    plt.scatter(centers[:, 0], centers[:, 1], c=centers_color, s=centers_s, alpha=centers_alpha)
    plt.show()
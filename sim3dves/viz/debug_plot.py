import matplotlib.pyplot as plt


class DebugPlot:
    def __init__(self, max_x, max_y):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.max_x = max_x
        self.max_y = max_y

    def render(self, entities):
        self.ax.clear()

        xs = []
        ys = []

        for e in entities:
            xs.append(e.position[0])
            ys.append(e.position[1])

        self.ax.scatter(xs, ys)
        self.ax.set_xlim(0, self.max_x)
        self.ax.set_ylim(0, self.max_y)

        plt.draw()
        plt.pause(0.001)

import matplotlib.pyplot as plt


class Plotter(object):
    def __init__(self, labels, colors, max_x):
        assert len(labels) == len(colors), \
            "labels and colors must be the same length"

        self.n_data = len(labels)
        self.labels = labels
        self.colors = colors
        self.datas = [[] for _ in range(self.n_data)]
        self.steps = []

        plt.ion()

        self.fig, self.axs = plt.subplots(
            1, 2, figsize=(10, 5),
            gridspec_kw={"width_ratios": [0.65, 0.35]}
        )
        self.pie = self.axs[1]
        self.line_plot = self.axs[0]
        self.lines = []
        for i in range(self.n_data):
            self.lines.append(
                self.line_plot.plot(
                    self.steps, self.datas[i],
                    label=self.labels[i],
                    color=self.colors[i]
                )[0]
            )
        self.line_plot.legend(
            bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.
        )
        self.line_plot.set_autoscaley_on(True)
        self.line_plot.set_autoscalex_on(True)

    def __del__(self):
        plt.close(self.fig)

    def reset(self):
        self.datas = [[] for _ in range(self.n_data)]
        self.steps = []

    def update(self, step, data):
        self.steps.append(step)
        for i, d in enumerate(data):
            self.datas[i].append(d)
            self.lines[i].set_xdata(self.steps)
            self.lines[i].set_ydata(self.datas[i])

        tot = sum(data)
        self.pie.clear()
        self.pie.pie([d / tot for d in data], colors=self.colors)

        self.line_plot.relim()
        self.line_plot.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

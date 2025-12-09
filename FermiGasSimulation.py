import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display

class FermiGas:
    def __init__(self):
        # Simulation Parameters
        self.N = 30
        self.M = 50
        self.kT = 2.0
        self.steps_per_frame = 200

        # System State
        self.state = np.zeros(self.M, dtype=bool)
        self.state[:self.N] = True

        self.occupancy_sum = np.zeros(self.M)
        self.samples = 0

        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.levels = np.arange(self.M)

        # Plot Elements
        self.bar_avg = self.ax.bar(self.levels, np.zeros(self.M), color='skyblue', alpha=0.6, label='Average')
        self.scat_inst = self.ax.scatter([], [], color='red', s=30, zorder=5, label='Instantaneous')
        self.line_theory, = self.ax.plot([], [], 'r--', linewidth=2, label='Theory')

        self.ax.set_xlim(-1, self.M)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_title("Fermi-Dirac Simulation (Colab)")
        self.ax.legend(loc='upper right')

        # IPYWidgets for Control
        self.slider_temp = widgets.FloatSlider(value=2.0, min=0.1, max=10.0, step=0.1, description='Temp (kT):')
        self.slider_N = widgets.IntSlider(value=30, min=5, max=45, step=1, description='Particles:')

        # Link widgets to update function
        self.slider_temp.observe(self.update_params, names='value')
        self.slider_N.observe(self.reset_system, names='value')

        # Layout
        self.ui = widgets.VBox([self.slider_temp, self.slider_N])

    def energy(self, level_index):
        return level_index * 1.0

    def step_simulation(self):
        for _ in range(self.steps_per_frame):
            occupied = np.where(self.state)[0]
            if len(occupied) == 0: continue

            old_idx = np.random.choice(occupied)
            new_idx = np.random.randint(0, self.M)

            if self.state[new_idx] and old_idx != new_idx: continue

            dE = self.energy(new_idx) - self.energy(old_idx)

            accept = False
            if dE <= 0:
                accept = True
            elif np.random.random() < np.exp(-dE / self.kT):
                accept = True

            if accept:
                self.state[old_idx] = False
                self.state[new_idx] = True

        self.occupancy_sum += self.state.astype(int)
        self.samples += 1

    def update_params(self, change):
        self.kT = change['new']
        self.occupancy_sum = np.zeros(self.M)
        self.samples = 0

    def reset_system(self, change):
        self.N = change['new']
        self.state = np.zeros(self.M, dtype=bool)
        self.state[:self.N] = True
        self.occupancy_sum = np.zeros(self.M)
        self.samples = 0

    def get_theory_curve(self):
        mu_min, mu_max = -10, self.M + 10
        for _ in range(20):
            mu = (mu_min + mu_max) / 2
            n_pred = np.sum(1 / (np.exp((self.levels - mu)/self.kT) + 1))
            if n_pred > self.N: mu_max = mu
            else: mu_min = mu
        return 1 / (np.exp((self.levels - mu)/self.kT) + 1)

    def animate(self, frame):
        self.step_simulation()

        avg = self.occupancy_sum / max(1, self.samples)
        for rect, h in zip(self.bar_avg, avg): rect.set_height(h)

        occupied_indices = np.where(self.state)[0]
        self.scat_inst.set_offsets(np.c_[occupied_indices, np.ones_like(occupied_indices)*0.5])

        theory = self.get_theory_curve()
        self.line_theory.set_data(self.levels, theory)

        return self.bar_avg, self.scat_inst, self.line_theory

# Run in Colab
sim = FermiGas()
display(sim.ui) # Show sliders
ani = FuncAnimation(sim.fig, sim.animate, interval=50, blit=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

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
        # Adjust bottom to make room for sliders
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        self.levels = np.arange(self.M)

        # Plot Elements
        self.bar_avg = self.ax.bar(self.levels, np.zeros(self.M), color='skyblue', alpha=0.6, label='Average')
        self.scat_inst = self.ax.scatter([], [], color='red', s=30, zorder=5, label='Instantaneous')
        self.line_theory, = self.ax.plot([], [], 'r--', linewidth=2, label='Theory')

        self.ax.set_xlim(-1, self.M)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xlabel("Energy Level")
        self.ax.set_ylabel("Occupancy Probability")
        self.ax.set_title("Fermi-Dirac Simulation")
        self.ax.legend(loc='upper right')

        # --- Matplotlib Widgets (Sliders) ---

        # Define axes areas for the sliders [left, bottom, width, height]
        ax_temp = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_N = plt.axes([0.25, 0.05, 0.65, 0.03])

        # Create Sliders
        self.slider_temp = Slider(
            ax=ax_temp,
            label='Temp (kT)',
            valmin=0.1,
            valmax=10.0,
            valinit=self.kT,
            valstep=0.1
        )

        self.slider_N = Slider(
            ax=ax_N,
            label='Particles',
            valmin=5,
            valmax=45,
            valinit=self.N,
            valstep=1
        )

        # Link sliders to update functions
        self.slider_temp.on_changed(self.update_temp)
        self.slider_N.on_changed(self.update_N)

    def energy(self, level_index):
        return level_index * 1.0

    def step_simulation(self):
        for _ in range(self.steps_per_frame):
            occupied = np.where(self.state)[0]
            if len(occupied) == 0: continue

            old_idx = np.random.choice(occupied)
            new_idx = np.random.randint(0, self.M)

            # Pauli Exclusion Principle check
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

    # Callback for Temperature Slider
    def update_temp(self, val):
        self.kT = val
        # Reset statistics on parameter change so the average adapts quickly
        self.occupancy_sum = np.zeros(self.M)
        self.samples = 0

    # Callback for Particle Count Slider
    def update_N(self, val):
        self.N = int(val)
        # Reset system state completely
        self.state = np.zeros(self.M, dtype=bool)
        self.state[:self.N] = True
        self.occupancy_sum = np.zeros(self.M)
        self.samples = 0

    def get_theory_curve(self):
        # Calculate Chemical Potential (mu) numerically based on particle count N
        mu_min, mu_max = -10, self.M + 10
        for _ in range(20): # Binary search for mu
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
        # Update scatter plot (x=index, y=0.5 constant)
        if len(occupied_indices) > 0:
            self.scat_inst.set_offsets(np.c_[occupied_indices, np.ones_like(occupied_indices)*0.5])
        else:
            self.scat_inst.set_offsets(np.zeros((0, 2)))

        theory = self.get_theory_curve()
        self.line_theory.set_data(self.levels, theory)

        return self.bar_avg.patches + [self.scat_inst, self.line_theory]

if __name__ == "__main__":
    sim = FermiGas()
    # Note: We must keep a reference to 'ani' (animation object)
    ani = FuncAnimation(sim.fig, sim.animate, interval=50, blit=True)
    plt.show()

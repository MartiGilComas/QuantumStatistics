import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import poisson

# ==========================================
# CONFIGURATION
# ==========================================
AVG_PHOTONS = 10         # Average brightness (Higher = easier to see distributions)
N_SCATTERERS = 100       # Number of random waves for thermal sum
HISTORY_LEN = 50         # How many previous phasors to show as a 'trail'
FRAMES = 500             # Number of trials to animate
INTERVAL = 20            # Speed (ms between frames)

class LightExperiment:
    def __init__(self, avg_photons, n_scatterers):
        self.avg_photons = avg_photons
        self.n_scatterers = n_scatterers

        # Theoretical Probabilities for overlays
        self.max_k = int(avg_photons * 3.5)
        self.k_axis = np.arange(self.max_k)

        # Poisson (Coherent)
        self.p_poisson = poisson.pmf(self.k_axis, avg_photons)

        # Bose-Einstein (Thermal) - Geometric Distribution
        denom = 1 + avg_photons
        self.p_bose = (1/denom) * (avg_photons/denom)**self.k_axis

    def get_coherent_trial(self):
        """Coherent: Constant E-field, Random Poisson Count"""
        E_field = np.sqrt(self.avg_photons) + 0j
        # Intensity is constant
        intensity = np.abs(E_field)**2
        # Count fluctuates (Shot Noise)
        count = np.random.poisson(intensity)
        return E_field, intensity, count

    def get_thermal_trial(self):
        """Thermal: Random Walk E-field, Random Poisson Count"""
        phases = np.random.uniform(0, 2*np.pi, self.n_scatterers)
        E_total = np.sum(np.exp(1j * phases))

        # Normalize to target average intensity
        scale = np.sqrt(self.avg_photons / self.n_scatterers)
        E_final = E_total * scale

        # Intensity fluctuates (Wave Interference)
        intensity = np.abs(E_final)**2
        # Count fluctuates (Shot Noise + Intensity Noise)
        count = np.random.poisson(intensity)
        return E_final, intensity, count

# ==========================================
# VISUALIZATION
# ==========================================
def run_animation():
    sim = LightExperiment(AVG_PHOTONS, N_SCATTERERS)

    # Setup Figure: 2 Rows (Coherent, Thermal) x 3 Columns
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # --- AXES DEFINITIONS ---
    ax_coh_phasor = fig.add_subplot(gs[0, 0])
    ax_coh_meter  = fig.add_subplot(gs[0, 1])
    ax_coh_hist   = fig.add_subplot(gs[0, 2])

    ax_th_phasor  = fig.add_subplot(gs[1, 0])
    ax_th_meter   = fig.add_subplot(gs[1, 1])
    ax_th_hist    = fig.add_subplot(gs[1, 2])

    # Storage for histograms
    coh_counts = []
    th_counts = []

    # Storage for thermal phasor trail
    th_trail_real = []
    th_trail_imag = []

    # --- SETUP HELPER FUNCTIONS ---
    def setup_phasor(ax, title, color):
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        limit = 3.5 * np.sqrt(AVG_PHOTONS)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', alpha=0.2)
        ax.axvline(0, color='gray', alpha=0.2)
        ax.set_xlabel("Re(E)")
        ax.set_ylabel("Im(E)")

    def setup_meter(ax, title, color):
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, AVG_PHOTONS * 3)
        ax.set_xticks([])
        ax.set_ylabel("Measured Count (n)")

    def setup_hist(ax, title, theory_curve, color):
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("Count (n)")
        ax.set_xlim(0, sim.max_k)
        ax.set_ylim(0, max(theory_curve) * 1.4)
        ax.plot(sim.k_axis, theory_curve, color='k', linestyle='--', linewidth=1.5, label='Theory', alpha=0.6)
        ax.legend(loc='upper right', frameon=False, fontsize='small')

    # Apply Setup
    setup_phasor(ax_coh_phasor, "Coherent Source (Laser)", 'navy')
    setup_meter(ax_coh_meter,   "Detector Output", 'navy')
    setup_hist(ax_coh_hist,     "Statistics (Poisson)", sim.p_poisson, 'navy')

    setup_phasor(ax_th_phasor,  "Thermal Source (Speckle)", 'darkred')
    setup_meter(ax_th_meter,    "Detector Output", 'darkred')
    setup_hist(ax_th_hist,      "Statistics (Bose-Einstein)", sim.p_bose, 'darkred')

    # --- PLOT OBJECTS (Initialized once) ---
    # Coherent
    coh_bar = ax_coh_meter.bar([0.5], [0], width=0.4, color='navy', alpha=0.7)[0]
    coh_text = ax_coh_meter.text(0.5, 0, "", ha='center', va='bottom', fontweight='bold', color='navy')

    # Thermal
    th_bar = ax_th_meter.bar([0.5], [0], width=0.4, color='darkred', alpha=0.7)[0]
    th_text = ax_th_meter.text(0.5, 0, "", ha='center', va='bottom', fontweight='bold', color='darkred')
    trail_line, = ax_th_phasor.plot([], [], '.', color='darkred', markersize=2, alpha=0.3)

    def update(frame):
        # 1. GENERATE DATA
        E_c, I_c, k_c = sim.get_coherent_trial()
        E_t, I_t, k_t = sim.get_thermal_trial()

        coh_counts.append(k_c)
        th_counts.append(k_t)

        # 2. UPDATE COHERENT (Blue)
        # Phasor arrow (remove old, add new)
        for p in list(ax_coh_phasor.patches): p.remove()
        ax_coh_phasor.arrow(0, 0, E_c.real, E_c.imag, head_width=0.4, color='navy')

        # Meter (Showing COUNT n, not Intensity I)
        coh_bar.set_height(k_c)
        coh_text.set_text(f"{k_c}")
        coh_text.set_y(k_c + 0.5)

        # Histogram
        for p in list(ax_coh_hist.patches): p.remove()
        if len(coh_counts) > 1:
            bins = np.arange(0, max(coh_counts)+2) - 0.5
            ax_coh_hist.hist(coh_counts, bins=bins, density=True, color='navy', alpha=0.3, rwidth=0.8)

        # 3. UPDATE THERMAL (Red)
        # Phasor arrow
        for p in list(ax_th_phasor.patches): p.remove()
        ax_th_phasor.arrow(0, 0, E_t.real, E_t.imag, head_width=0.4, color='darkred')

        # Trail
        th_trail_real.append(E_t.real)
        th_trail_imag.append(E_t.imag)
        if len(th_trail_real) > HISTORY_LEN:
            th_trail_real.pop(0)
            th_trail_imag.pop(0)
        trail_line.set_data(th_trail_real, th_trail_imag)

        # Meter
        th_bar.set_height(k_t)
        th_text.set_text(f"{k_t}")
        th_text.set_y(k_t + 0.5)

        # Histogram
        for p in list(ax_th_hist.patches): p.remove()
        if len(th_counts) > 1:
            bins = np.arange(0, max(th_counts)+2) - 0.5
            ax_th_hist.hist(th_counts, bins=bins, density=True, color='darkred', alpha=0.3, rwidth=0.8)

        return []

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_animation()

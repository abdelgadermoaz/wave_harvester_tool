# examples/example_single_patch.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make sure we can import the package if running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harvester.models import PiezoParams, simulate_piezo
from harvester.vibration import read_vibration_csv, preprocess_vibration


def main():
    # 1) Load base vibration
    t, a = read_vibration_csv(os.path.join(ROOT, "data", "sample_vibration.csv"))

    # Optional preprocessing
    t, a = preprocess_vibration(t, a, f_low=1.0, f_high=50.0)

    # 2) Define piezo harvester parameters (dummy but realistic-ish)
    params = PiezoParams(
        m=0.02,        # 20 g
        c=0.063,       # small damping
        k=20.0,        # N/m
        Cp=1e-7,       # 1 µF
        theta=1e-4,    # coupling
        R=1e5,         # 100 kOhm
    )

    # 3) Run simulation
    result = simulate_piezo(t, a, params)

    print(f"Average power: {result['P_avg']:.6f} W")

    # 4) Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, a)
    axs[0].set_ylabel("Base accel [m/s²]")
    axs[0].grid(True)

    axs[1].plot(t, result["v"])
    axs[1].set_ylabel("Voltage [V]")
    axs[1].grid(True)

    axs[2].plot(t, result["P"])
    axs[2].set_ylabel("Power [W]")
    axs[2].set_xlabel("Time [s]")
    axs[2].grid(True)

    fig.suptitle("Single Piezo Harvester Response")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

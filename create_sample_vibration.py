import numpy as np
import pandas as pd
import os

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

t_end = 10.0       # total duration, seconds
dt = 0.001         # time step, seconds
t = np.arange(0.0, t_end + dt, dt)

f = 5.0            # frequency in Hz
a0 = 1.0           # amplitude in m/s^2

a = a0 * np.sin(2 * np.pi * f * t)

df = pd.DataFrame({"t": t, "a_hull": a})
out_path = os.path.join("data", "sample_vibration.csv")
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
A = 125000         # Amplitude
alpha = 64000      # Exponential decay rate
omega = 86.13e3    # Angular frequency in radians per second

# Define time array
fs = 1e6           # Sampling frequency (1 MHz)
t = np.arange(0, 0.001, 1/fs)  # Time vector from 0 to 1 ms with step of 1/fs

# Define the signal i(t)
i_t = A * np.exp(-alpha * t) * np.sin(omega * t)

# Compute the DFT
I_f = np.fft.fft(i_t)

# Frequency vector
N = len(i_t)          # Number of samples
freq = np.fft.fftfreq(N, d=1/fs)  # Frequency bins

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot time domain signal
plt.subplot(2, 1, 1)
plt.plot(t, i_t)
plt.title('Time Domain Signal i(t)')
plt.xlabel('Time (s)')
plt.ylabel('i(t)')
plt.grid()

# Plot frequency domain (magnitude spectrum)
plt.subplot(2, 1, 2)
plt.plot(freq[:N // 2], np.abs(I_f[:N // 2]))  # Plot positive frequencies only
plt.title('Magnitude Spectrum of i(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()

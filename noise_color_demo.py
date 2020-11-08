import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import sounddevice as sd


def generate_signal_from_spectrum(C_k: np.ndarray)-> np.ndarray:
    """
    generates a real valued time domain signal from frequency coefficients for ifft. Signal is aprox. scaled to ~[-1;1].
    :param C_k: np.ndarray, dtype='float', one dimension: frequency coefficients, two sided. Which index is representing
    which frequency is like in scipy.fft.fftfreq
    :return: np.ndarray, dtype='float': time domain signal ~[-1;1] according to specified C_k
    """
    length = C_k.shape[0]
    # generate uniform distributed phase values to get real valued signal
    phi_k = np.random.uniform(0, 2 * np.pi, length)
    # calculate complex coefficients to get real signal
    C_k_complex = np.ndarray(length, dtype="complex")
    for k in range(-((length // 2)+1), (length // 2)):
        C_k_complex[k] = C_k[k] * np.exp(1j * phi_k[k] * k)
    # get signal by inverse fft, take only the the real part
    signal = sc.fft.ifft(C_k_complex).real
    # normalise signal
    return signal / max(np.abs(signal.min()), signal.max())


def generate_pink_noise(duration: float, f_sample: float) -> np.ndarray:
    """
    generates a real valued time domain signal of pink noise. Signal is aprox. scaled to ~[-1;1].
    :param duration: duration the noise signal should have in seconds
    :param f_sample: sample frequency, determines the frequency/time resolution
    :return: np.ndarray, dtype='float': real valued time domain signal ~[-1;1] of pink noise
    """
    # calculate the number of time points and the time difference between them
    length = duration * f_sample
    delta_t = 1 / f_sample
    # get the angular frequency axis in [omega]
    f_k = sc.fft.fftfreq(length, d=delta_t)/(2 * np.pi)
    # calculate the frequency coefficients based on the noise color
    # pink noise has power spectral density of 1/f -> the amplitude has 1/sqrt(f)
    C_k = np.ndarray(length)
    C_k[1:] = 1 / np.sqrt(np.abs(f_k[1:]))
    # no dc
    C_k[0] = 0
    # generate the signal from the frequency coefficients
    return generate_signal_from_spectrum(C_k)


def generate_brown_noise(duration: float, f_sample: float) -> np.ndarray:
    """
    generates a real valued time domain signal of brown/red noise. Signal is aprox. scaled to ~[-1;1].
    :param duration: duration the noise signal should have in seconds
    :param f_sample: sample frequency, determines the frequency/time resolution
    :return: np.ndarray, dtype='float': real valued time domain signal ~[-1;1] of brown/red noise
    """
    # calculate the number of time points and the time difference between them
    length = duration * f_sample
    delta_t = 1 / f_sample
    # get the angular frequency axis in [omega]
    # calculate the frequency coefficients based on the noise color
    # brown/red noise has power spectral density of 1/(f**2) -> the amplitude has 1/f
    f_k = sc.fft.fftfreq(length, d=delta_t) / (2 * np.pi)
    C_k = np.ndarray(length)
    C_k[1:] = 1 / np.abs(f_k[1:])
    # no dc
    C_k[0] = 0
    # generate the signal from the frequency coefficients
    return generate_signal_from_spectrum(C_k)

def generate_infrared_noise(duration: float, f_sample: float) -> np.ndarray:
    """
    generates a real valued time domain signal of 'infrared' noise. Signal is aprox. scaled to ~[-1;1].
    :param duration: duration the noise signal should have in seconds
    :param f_sample: sample frequency, determines the frequency/time resolution
    :return: np.ndarray, dtype='float': real valued time domain signal ~[-1;1] of 'infrared' noise
    """
    # calculate the number of time points and the time difference between them
    length = duration * f_sample
    delta_t = 1 / f_sample
    # get the angular frequency axis in [omega]
    # calculate the frequency coefficients based on the noise color
    # 'infrared' noise has power spectral density of 1/(f**4) -> the amplitude has 1/(f**2)
    f_k = sc.fft.fftfreq(length, d=delta_t) / (2 * np.pi)
    C_k = np.ndarray(length)
    C_k[1:] = 1 / (np.abs(f_k[1:])**2)
    # no dc
    C_k[0] = 0
    # generate the signal from the frequency coefficients
    return generate_signal_from_spectrum(C_k)

def generate_white_noise(duration: float, f_sample: float) -> np.ndarray:
    """
    generates a real valued time domain signal of white noise. Signal is aprox. scaled to ~[-1;1].
    :param duration: duration the noise signal should have in seconds
    :param f_sample: sample frequency, determines the frequency/time resolution
    :return: np.ndarray, dtype='float': real valued time domain signal ~[-1;1] of white noise
    """
    # calculate the number of time points and the time difference between them
    length = duration * f_sample
    delta_t = 1 / f_sample
    # calculate the frequency coefficients based on the noise color
    # white noise has constant power spectral density over the frequency
    C_k = np.ones(length)
    # no dc
    C_k[0] = 0
    # generate the signal from the frequency coefficients
    return generate_signal_from_spectrum(C_k)


# sampling frequency
fs = 44500  # < [Hz]
# duration of signal
dur_signal = 3  # < [s]
# length of the time domain signal displayed
dur_snipped = 2# < [s]

# calculate the number of points
signal_len = int(fs * dur_signal)
snipped_len = int(fs * dur_snipped)

# set up the plot
fig = plt.figure(figsize=[15, 7])
sub = fig.subplots(1, 2)

# first generate, plot and play pink white
noise_color = "white"

sub[0].clear()
sub[0].set_title(f"{noise_color} noise: time domain signal")
sub[0].set_xlabel("time(s)")
sub[0].set_ylabel("amplitude")

sub[1].clear()
sub[1].set_title(f"{noise_color} noise: frequency domain signal")
sub[1].set_ylabel("power [dB]")
sub[1].set_xlabel("frequency [Hz]")
sub[1].set_xscale('log')

noise_signal = generate_white_noise(dur_signal, fs)
# time domain subplot with duration of {dur_snipped}
sub[0].plot([i / fs for i in range(snipped_len)], noise_signal[:int(dur_snipped * fs)])
# calculate the power spectral density of the noise
# calculate the fft of the signal
noise_psd = sc.fft.fft(noise_signal)
# only take the positive half side and only the amount of the complex values
noise_psd = np.abs(noise_psd[:signal_len // 2])
# calculate the decibel value psd[dB] = 10* log10(amplitude[lin]**2) = 20* log10(amplitude[lin])
noise_psd = 20 * np.log10(noise_psd)

sub[1].plot(np.linspace(0, fs/2, signal_len//2)[1:], noise_psd[1:])
# bg_thread = threading.Thread(target=background_task, args=[noise_signal, fs])
# bg_thread.start()

plt.show()
plt.pause(0.1)

# bg_thread.join()
# plot_thread = multiprocessing.Process(target=plot_task, args=[noise_color, noise_signal, noise_psd])
# plot_thread.start()
while True:
    ans = input("for playing the noise sound press p, for coniuing to the next noise press c")
    if len(ans) == 1:
        if ans.count("c") != 0:
            break
        if ans.count("p") != 0:
            sd.play(noise_signal, fs)



# pink noise
noise_color = "pink"

sub[0].clear()
sub[0].set_title(f"{noise_color} noise: time domain signal")
sub[0].set_xlabel("time(s)")
sub[0].set_ylabel("amplitude")

sub[1].clear()
sub[1].set_title(f"{noise_color} noise: frequency domain signal")
sub[1].set_ylabel("power [dB]")
sub[1].set_xlabel("frequency [Hz]")
sub[1].set_xscale('log')

noise_signal = generate_pink_noise(dur_signal, fs)
# time domain subplot with duration of {dur_snipped}
sub[0].plot([i / fs for i in range(snipped_len)], noise_signal[:int(dur_snipped * fs)])
# calculate the power spectral density of the noise
# calculate the fft of the signal
noise_psd = sc.fft.fft(noise_signal)
# only take the positive half side and only the amount of the complex values
noise_psd = np.abs(noise_psd[:signal_len // 2])
# calculate the decibel value psd[dB] = 10* log10(amplitude[lin]**2) = 20* log10(amplitude[lin])
noise_psd = 20 * np.log10(noise_psd)

sub[1].plot(np.linspace(0, fs/2, signal_len//2)[1:], noise_psd[1:])
plt.show()
plt.pause(0.1)

while True:
    ans = input("for playing the noise sound press p, for coniuing to the next noise press c")
    if len(ans) == 1:
        if ans.count("c") != 0:
            break
        if ans.count("p") != 0:
            sd.play(noise_signal, fs)

# brown noise
noise_color = "brown"

sub[0].clear()
sub[0].set_title(f"{noise_color} noise: time domain signal")
sub[0].set_xlabel("time(s)")
sub[0].set_ylabel("amplitude")

sub[1].clear()
sub[1].set_title(f"{noise_color} noise: frequency domain signal")
sub[1].set_ylabel("power [dB]")
sub[1].set_xlabel("frequency [Hz]")
sub[1].set_xscale('log')

noise_signal = generate_brown_noise(dur_signal, fs)
# time domain subplot with duration of {dur_snipped}
sub[0].plot([i / fs for i in range(snipped_len)], noise_signal[:int(dur_snipped * fs)])
# calculate the power spectral density of the noise
# calculate the fft of the signal
noise_psd = sc.fft.fft(noise_signal)
# only take the positive half side and only the amount of the complex values
noise_psd = np.abs(noise_psd[:signal_len // 2])
# calculate the decibel value psd[dB] = 10* log10(amplitude[lin]**2) = 20* log10(amplitude[lin])
noise_psd = 20 * np.log10(noise_psd)

sub[1].plot(np.linspace(0, fs/2, signal_len//2)[1:], noise_psd[1:])
plt.show(block=False)
plt.pause(0.1)
while True:
    ans = input("for playing the noise sound press p, for coniuing to the next noise press c")
    if len(ans) == 1:
        if ans.count("c") != 0:
            break
        if ans.count("p") != 0:
            sd.play(noise_signal, fs)


# brown noise
noise_color = "infrared"

sub[0].clear()
sub[0].set_title(f"{noise_color} noise: time domain signal")
sub[0].set_xlabel("time(s)")
sub[0].set_ylabel("amplitude")

sub[1].clear()
sub[1].set_title(f"{noise_color} noise: frequency domain signal")
sub[1].set_ylabel("power [dB]")
sub[1].set_xlabel("frequency [Hz]")
sub[1].set_xscale('log')

noise_signal = generate_infrared_noise(dur_signal, fs)
# time domain subplot with duration of {dur_snipped}
sub[0].plot([i / fs for i in range(snipped_len)], noise_signal[:int(dur_snipped * fs)])
# calculate the power spectral density of the noise
# calculate the fft of the signal
noise_psd = sc.fft.fft(noise_signal)
# only take the positive half side and only the amount of the complex values
noise_psd = np.abs(noise_psd[:signal_len // 2])
# calculate the decibel value psd[dB] = 10* log10(amplitude[lin]**2) = 20* log10(amplitude[lin])
noise_psd = 20 * np.log10(noise_psd)

sub[1].plot(np.linspace(0, fs/2, signal_len//2)[1:], noise_psd[1:])
plt.show(block=False)
plt.pause(0.1)
while True:
    ans = input("for playing the noise sound press p, for coniuing to the next noise press c")
    if len(ans) == 1:
        if ans.count("c") != 0:
            break
        if ans.count("p") != 0:
            sd.play(noise_signal, fs)


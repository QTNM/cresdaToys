# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Back of the envelope frequency resolution
#
# ### Cyclotron radiation
# The angular cyclotron frequency, $\Omega_c$, of ana lectron with kinetic energy $T$ and mass $m_{e}$ in a magnetic field $B$ is
# $$ \Omega_{C} = \frac{eB}{\gamma m_e}=\frac{eB}{m_e + T/c^2} $$
# The cyclotron frequency $f_c$ is related to the angular cyclotron frequency in the usual fashion.
# $$ f_c = \frac{\Omega_c}{2 \pi} $$
#

# Import some standard python analysis and plotting packages
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# Define the kinetic energy at the tritium end point
T_end=18.6*1000*scipy.constants.e  #18.6keV in J
B=1 # 1 T

print(T_end)


def getCyclotronFreq(B,T):
    """Function to return the cyclotron frequency emitted by an electron with kinetic energy T 
    in a magentic field B

    Args:
        B: The magentic field in Tesla
        T: The kinetic energy in Joules
        
    Returns:
        The cylotron frequency, f_c, in Hz.

    """
    return ((scipy.constants.e*B)/(scipy.constants.m_e + T/(scipy.constants.c**2)))/(2*np.pi)


# Plot the cyclotron frequency near the end point
fig,ax = plt.subplots() 
tArray = np.linspace(0.9*T_end,T_end,100)  # 100 points from 0.9 *T_end to T_end in an numpy array
omegaArray= getCyclotronFreq(B,tArray) # Evaluate the cyclotron frequency for these energies
ax.plot(tArray/scipy.constants.e,omegaArray/1e9)  #Conversion to eV and GHz
ax.set_xlabel("Electron Kinetic Energy (eV)")
ax.set_ylabel("Cyclotron Frequency (GhZ)")
ax.set_title("Electron K.E vs Cyclotron Frequency")

# #### Zooming right into the end point

fig,ax = plt.subplots()
tArray = np.linspace(0.9999*T_end,T_end,100)
omegaArray= getCyclotronFreq(B,tArray)
ax.plot((tArray/scipy.constants.e)-18.6e3,(omegaArray-27.0e9)/1e6)  #Conversion to eV and GHz
ax.set_xlabel("Electron Kinetic Energy -- Difference to 18.6keV (eV)")
ax.set_ylabel("Cyclotron Frequency [-27GHz] (MhZ)")
ax.set_title("Energy Difference to 18.6keV")
#_ = plt.xticks(rotation=60)


# From this plot we see that the cyclotron frequency changes by $\Delta f \approx50$kHz per electron volt of kinetic energy. So we should keep that in mind for our goal frequency resolution.
#
# The time scale associated with resolving that frequency difference is:
# $$ T_{resolve} \approx \frac{1}{\Delta f} = 20 \mu\text{s}$$

# ## How long do we need to record a signal for to get energy resolution XXX?
#
# There are a few paramters that might effect our energy resolution from a first principles perspective. Let's make our world simple and suppose we are just trying to record the frequency of a 27GHz sine wave.
#
# Rather than investing in a 100Ghz digitiser (which doesn't exist as far as I know) we instead have to shift the frequency to a lower level.
#
# ### Aside: Mixer Theory
#
# Suppose we have an RF signal which is at a fixed frequency, $\omega_0$:
# $$ v_{RF} = A(t) \cos \left( \omega_0 t + \phi(t) \right) $$
#
# We are going to mix with a local oscillator with frequency, $\omega_{LO}$
# $$v_{LO} = A_{LO} \cos \left( \omega_{LO} t \right) $$
#
# If we mix these together we get
# $$v_{out} = v_{RF} \times v_{LO} $$
# $$ v_{out} = A(t)A_{LO} \cos \left( \omega_0 t + \phi(t) \right) \cos \left( \omega_{LO} t \right) $$
#
# Now remember
# $$ \cos(A+B) = \cos A \cos B - \sin A \sin B $$
# so
# $$ \cos A \cos B = \frac{1}{2} \left[ \cos(A+B) + \cos (A-B) \right] $$
#
# Therefore
# $$  v_{out} = \frac{A(t)A_{LO}}{2} \left[ \cos \left( \omega_0 t + \phi(t) + \omega_{LO} t \right) + \cos \left( \omega_0 t + \phi(t) - \omega_{LO} t \right) \right] $$
# $$ v_{out} = \frac{A(t)A_{LO}}{2} \left[ \cos \left( (\omega_0 +\omega_{LO}) t + \phi(t)\right) + \cos \left( (\omega_0 -\omega_{LO}) t + \phi(t)\right) \right] $$
#
# The signal is mixed to $f_{LO}+f_{RF}$ and $f_{LO}-f_{RF}$. Now we have to be careful about which local oscillator frequency we choose and realise that $f_{RF}=f_{LO} \pm f_{IF}$ and $f_{IM}=f_{LO} \mp f_{IF}$ both end up at $f_{IF}$ 
#
# Just for the sake of argument lets pick our mixing frequency to be 26.5GHz and lets operate our system with a low pass filter that gets rid of the high-frequency band of the mixer.
#
# So our mixed signal will be
# $$ v_{out} = \frac{A(t)A_{LO}}{2}  \cos \left( (\omega_0 -\omega_{LO}) t + \phi(t)\right) $$

# ## Best case scenario $A(t)=A$ and $\phi(t)=\phi$ 
# The best case scenario is the one where the amplitude and phase are constant so then we can investigate how does our frequency resolution depend on observation time.
#
# The only other variable is the sampling rate.
#
# So we will look at $f_{centre}=60.46$MHz and a frequency change of $\Delta f=50kHz$. The conclusion will be that (for sampling rates above Nyquist (of the centre frequency) our frequency resolving resolution does not depend on sampling rates. It only depends on the resolving time. 

# +
def getV(t,w,A,phi):
    """Function to return a simple cosine 

    Args:
        t: The array of times
        w: The angular frequency
        A: The amplitude
        phi: The phase at t=0
 
     Returns:
        The cosine values as an array

    """
    return A*np.cos((w*t)+phi)

#Define the centre frequency and deltaf we need to measure
centref=60.46e6
deltaf=50e3
t_res=1./deltaf  #Our resolving time
dt=1e-9 #Our time between samples (1./sampling rate)

duration=t_res*1  #The time we will take as signal duration
#If duration >= t_res we can resolve deltaf, otherwise we can not

N=int(duration/dt)
w=2*np.pi*(centref) #60.46 MHz
w2=2*np.pi*(centref+deltaf) #60.51 MHz
A=1 #for simplicity
phi=0 # for simplicity
sr=1./dt

print("Sampling rate:",sr,"Hz")
print("Time between samples:",dt,"s")
print("Number of samples:",N)
print("Signal duration",duration,"s")


t=np.linspace(0,dt*N,N)
v=getV(t,w,A,phi)
v2=getV(t,w2,A,phi)

fig,ax = plt.subplots()
ax.plot(t[0:100],v[0:100],label="f="+str(w/(2*np.pi)/1e6)+"MHz")
ax.plot(t[0:100],v2[0:100],label="f="+str(w2/(2*np.pi)/1e6)+"MHz")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (arb)")
ax.legend()


# +
def convertToMag(yf):
    """Function to convert the result of an fft to magnitude

    Args:
        yf: The array of complex numbers from the fft (this will be both the positive and negative frequencies)
 
     Returns:
        The array of magnitudes (of length N/2) where N is the length of yf

    """
    N=yf.shape[0] #The length of yf
    return 2.0/N * np.abs(yf[0:N//2])  # The 2/N is a normalisation

from scipy.fft import fft, ifft  # Import the fft and inverse fft functions
yf = fft(v) #FFT of v
yf2 = fft(v2) #FFT of v2
df=1/(N*dt)  # Frequency spacing
xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)  #The N/2 frequency values from 0 to 1/(2*dt)

#Plot the frequency around the centre frequency bin
fig,ax = plt.subplots()
fbin=int(centref/df)
ax.plot((xf/1e6)[fbin-20:fbin+20],convertToMag(yf)[fbin-20:fbin+20],label="f="+str(w/(2*np.pi)/1e6)+"MHz") #1e6 to convert to MHz
ax.plot((xf/1e6)[fbin-20:fbin+20],convertToMag(yf2)[fbin-20:fbin+20],label="f="+str(w2/(2*np.pi)/1e6)+"MHz")  #1e6 to convert to MHz
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Magnitude (arb.)")
ax.set_title("Frequency Resolving Power at duration="+str(duration)+"s")
ax.legend()

# -

print((xf[1]-xf[0])/1e6)
print(df/1e6)



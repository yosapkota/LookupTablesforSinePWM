import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

a = 2/math.sqrt(3)

start_value = 137500
positive_peak = 150000
negative_peak = 125000
table_size = 3600 # this is to denote 360 degrees, actual table size is 3600
diff = positive_peak - start_value

SINE_VALUE = [0]*3600
THI_SINE_VALUE = [0]*3600

SPWM_OUT = [0]*3600
THI_SPWM_OUT = [0]*3600
SV_PWM_OUT = [0]*3600
sawtooth_counter = [0]*3600
sawtooth_counter[0] = negative_peak

phaseA = [0]*3600
phaseB = [0]*3600
phaseC = [0]*3600
instMin = []
SVPWM_PhaseA = [0]*3600
SVPWM_PhaseB = []
SVPWM_PhaseC = []
cmv = [0]*3600

lut = []
angles = []

i, j = 0,0
print(sawtooth_counter[0])
countdirection = 1 # for up and 0 for down counting
while(i < table_size):

    # reference signals - three phase sine wave
    sine_val = math.sin(math.radians(i/10))

    phaseA[i] = a*math.sin(math.radians(i/10))
    phaseB[i] = a*math.sin(math.radians(i/10) - 2*(math.pi)/3)
    phaseC[i] = a*math.sin(math.radians(i/10) - 4*(math.pi)/3)
    
    phaseAInstVal = phaseA[i]
    phaseBInstVal = phaseB[i]
    phaseCInstVal = phaseC[i]

    # print(sawtooth_counter[i])
    # Sawtooth counter
    if(sawtooth_counter[i] < positive_peak): #countdirection == 1):
        if(i == 0):
            sawtooth_counter[i] = negative_peak
        else:  
            sawtooth_counter[i] = int(sawtooth_counter[i-1] + 5000/12)
            # if(sawtooth_counter[i-1] > positive_peak):
            #     countdirection = 0
    if(sawtooth_counter[i] > positive_peak): # asymmetrical 
        sawtooth_counter[i] = negative_peak

    # if(countdirection == 0):
    #     sawtooth_counter[i] = int(sawtooth_counter[i-1] - 5000/12) #symmetrical carrier
    #     if(sawtooth_counter[i-1] < negative_peak):
    #             countdirection = 1
    
    
    
        
    # Offset adjusted phase A sine wave 
    SINE_VALUE[i] = start_value + int(sine_val * diff)

    # Offset adjusted Third harmnonic injected phase A sine value
    third_harmonic = (1/6) * math.sin(math.radians(3*i/10))
    val = start_value + int((a*sine_val+ third_harmonic) * diff)
    THI_SINE_VALUE[i] = val

    # Common mode voltage calculation for Space Vector PWM 
    # find the minimum of three phase for given i
    minVal = min(phaseAInstVal,phaseBInstVal,phaseCInstVal)
   # find the maximum of three phase for given i
    maxVal = max(phaseAInstVal,phaseBInstVal,phaseCInstVal)
    # common mode voltage
    cmv[i] = - ((minVal + maxVal)/2)


    valA = phaseA[i] + cmv[i]
    SVPWM_PhaseA[i] = valA

    phaseAoffset = start_value + int(valA * diff)
    SVPWM_PhaseA[i] = phaseAoffset

    
    lut.append(phaseAoffset)
    angles.append(j)

    # PWM generation
    if(SINE_VALUE[i] > sawtooth_counter[i]):
        SPWM_OUT[i] = 1
    else:
        SPWM_OUT[i] = 0
    
    if(THI_SINE_VALUE[i] > sawtooth_counter[i]):
        THI_SPWM_OUT[i] = 1
    else:
        THI_SPWM_OUT[i] = 0
    
    if(SVPWM_PhaseA[i] > sawtooth_counter[i]):
        SV_PWM_OUT[i] = 1
    else:
        SV_PWM_OUT[i] = 0

    i = i + 1
    j = j + 0.1

print(i,j)

# print(lut_hex)
# print(lut)
# lut_hex = [hex(val) for val in lut] 
# print(lut_hex)
# print(lookup_table_hex)
# print(lookup_table)

# plt.subplot(3,1,1)
plt.plot(angles, sawtooth_counter)
plt.plot(angles, SINE_VALUE, label = "Sinusoidal PWM")
plt.plot(angles, THI_SINE_VALUE, label = "Third Harmonic Injected Sine PWM")
plt.plot(angles, SVPWM_PhaseA, label = "Space Vector PWM")
plt.title('Sawtooth counter and Different PWM Topologyies')
plt.legend(loc="upper right")
plt.xlabel('Angle (degrees)')
plt.ylabel('Values')
plt.show()


plt.subplot(3,1,1)
plt.plot(angles, SPWM_OUT)
plt.title('Sine')
# plt.xlabel('Angle (degrees)')
plt.ylabel('Values')

plt.subplot(3,1,2)
plt.plot(angles, THI_SPWM_OUT)
plt.title('THI Sine')
# plt.xlabel('Angle (degrees)')
plt.ylabel('Values')
# plt.show()

plt.subplot(3,1,3)
plt.plot(angles, SV_PWM_OUT)
plt.title('SVPWM')
plt.xlabel('Angle (degrees)')
plt.ylabel('Values')
plt.show()



# ## FFT plot
# sr = 3600 # number of samples

# # removing dc offset
# # SPWM_OUT = SPWM_OUT - np.mean(SPWM_OUT)


# SPWM_OUT_FFT = fft(SPWM_OUT)
# N = len(SPWM_OUT_FFT)
# n = np.arange(N)
# T = N/sr
# freq = n/T 

# THI_SPWM_OUT_FFT = fft(THI_SPWM_OUT)
# SV_PWM_OUT_FFT = fft(SV_PWM_OUT)

# # plt.figure(figsize = (12, 6))
# plt.subplot(131)
# plt.stem(freq, np.abs(SPWM_OUT_FFT), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 500)

# plt.subplot(132)
# plt.stem(freq, np.abs(THI_SPWM_OUT_FFT), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 500)

# plt.subplot(133)
# plt.stem(freq, np.abs(SV_PWM_OUT_FFT), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 500)

# plt.show()
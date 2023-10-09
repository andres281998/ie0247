# Imports python libraries
import numpy as np
import random as rd
import wave
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import butter, lfilter, filtfilt #for filtering data
from statistics import stdev
sys.path.insert(1, r'./../functions') # add to pythonpath

# commands to create high-resolution figures with large labels
plt.rcParams['axes.labelsize'] = 16 # fontsize for figure labels
plt.rcParams['axes.titlesize'] = 18 # fontsize for figure titles
plt.rcParams['font.size'] = 14 # fontsize for figure numbers
plt.rcParams['lines.linewidth'] = 1.4 # line width for plotting

#Function that extracts the number of recording channels, sampling rate, time and signal
#variable is the path and filename of the .wav file
def ecg(variable):
    record = wave.open(variable, 'r') # load the data

    # Get the number of channels, sample rate, etc.
    numChannels = record.getnchannels() #number of channels
    numFrames = record.getnframes() #number of frames
    sampleRate = record.getframerate() #sampling rate
    sampleWidth = record.getsampwidth()

    # Get wave data
    dstr = record.readframes(numFrames * numChannels)
    waveData = np.frombuffer(dstr, np.int16)

    # Get time window
    timeECG = np.linspace(0, len(waveData)/sampleRate, num=len(waveData))
    
    return timeECG, waveData


### Arreglo con el nombre de los archivos.wav
nice = ["a0001","a0002","a0003","a0004","a0005","a0006","a0007","a0008","a0009","a0010","a0011","a0012","a0013","a0014","a0015"]

#Test data
"""
# Toda esta seccion se deja comentada ya que es la parte 1 del proyecto, si se quiere usar de nuevo solo es de quitar las tres comillas
# del inicio y del final

for i in range(0, len(nice)): #Este for muestra los graficos de cada señal
    timeECG, waveData = ecg(nice[i]+'.wav')
    # Plotting EMG signal
    plt.figure(figsize=(18,6))
    plt.xlabel(r'time (s)')
    plt.ylabel(r'voltage ($\mu$V)')
    plt.plot(timeECG,waveData, 'b')
    plt.title(nice[i])
    plt.show()
    # Calcular la amplitud máxima
    max_amplitude = np.max(waveData)
    # Imprimir el valor máximo
    print(nice[i]+'.wav '+"Amplitud Máxima:", max_amplitude)


#Obtaining data
timeECG, waveData = ecg("a0001.wav")

# Plotting EMG signal (one beat)
plt.figure(figsize=(18,6))
plt.xlabel(r'time (s)')
plt.ylabel(r'voltage ($\mu$V)')
plt.plot(timeECG,waveData, 'b')
plt.xlim(1.4,1.8)
plt.title("One beat")
plt.show()

# Plotting EMG signal (three beats)
plt.figure(figsize=(18,6))
plt.xlabel(r'time (s)')
plt.ylabel(r'voltage ($\mu$V)')
plt.plot(timeECG,waveData, 'b')
plt.xlim(1.4,2.6)
plt.title("Three beats")
plt.show()

# Plotting EMG signal (15 seconds)
plt.figure(figsize=(18,6))
plt.xlabel(r'time (s)')
plt.ylabel(r'voltage ($\mu$V)')
plt.plot(timeECG,waveData, 'b')
plt.xlim(0,15)
plt.title("15 seconds")
plt.show()

"""
"""
Algoritmo para detectar componentes de ECG.
La idea que subyace a este algoritmo es encontrar un buen registro de ECG.

Luego, detectar los valores máximos (picos R) en una determinada ventana de tiempo (en función de un umbral).

Posteriormente, en base a ciertos retardos de tiempo desde el pico R, obtendremos otras partes de la señal periódica.

Estas componentes pueden ser valores mínimos (Q y S) o obtendremos la derivada del tiempo y 
detectaremos cuando se produce un punto de inflexión (onda P y onda T).

Picos R
Según la detección del pico R, podremos calcular la frecuencia cardíaca y los intervalos R-R.

La siguiente función crea una matriz de valores que superan un cierto umbral. Luego, determina el valor máximo de esta matriz y 
lo agrega en el vector R. Y esto se repite hasta el final de la serie temporal.

"""

def detecta_maximos_locales(timeECG, waveData, threshold_ratio=0.7):

    #Si no se detectan todas las crestas R, disminuya el threshold_ratio.
    #Si se detectan componentes que no son crestas R (como ondas T), incremente el threshold_ratio.

    if len(timeECG) != len(waveData): #Genera un error si dos arreglos tienen longitudes diferentes
        raise Exception("The two arrays have different lengths.")
    
    interval = max(waveData) - min(waveData)
    threshold = threshold_ratio*interval + min(waveData)
    maxima = []
    maxima_indices = []
    mxs_indices = []
    banner = False
    
    for i in range(0, len(waveData)):
            
        if waveData[i] >= threshold:#Si se supera un valor umbral,
            # los índices y valores se guardan
            banner = True
            maxima_indices.append(i)
            maxima.append(waveData[i])
            
        elif banner == True and waveData[i] < threshold: #Si se cruza el valor del umbral
            #se guarda el índice del valor máximo en el array original
            index_local_max = maxima.index(max(maxima))
            mxs_indices.append(maxima_indices[index_local_max])
            maxima = []
            maxima_indices = []
            banner = False     

    return mxs_indices

def R_intervals(time_indices):
    length = len(time_indices)
    if length > 1 : #### Se crea un if para tomar en cuenta los casos en que el lengt es 1 o menor, ya que pueden suceder errores como division entre 0
        intervals = np.zeros(length-1)
        for i in range(0, length-1):
            intervals[i] = time_indices[i+1]-time_indices[i]
    else: ### si es menor que 1 se devuelve un arreglo vacio que representa mejor la situación en la que no se pueden calcular intervalos R-R debido a la falta de datos
        intervals = []
    return intervals




for i in range(0, len(nice)): ###Se agrega un for para crear todos los graficos de cada señal
    timeECG, waveData = ecg(nice[i]+'.wav') ###Se crean las variables para cada señal
    #Probar la función y trazar.
    mxs_indices = detecta_maximos_locales(timeECG, waveData)

    mean_bpm = 60*(len(mxs_indices)/(timeECG[-1]-timeECG[0]))
    xx = R_intervals(timeECG[mxs_indices])
    mean_rr = np.mean(xx)
    mean_bpm_from_rr = 60 / mean_rr ###
    
    
    plt.figure(figsize=(18,8))
    plt.xlabel('time (s)')
    plt.ylabel(r'voltage ($\mu$V)')
    plt.xlim(min(timeECG),max(timeECG))
    plt.plot(timeECG, waveData)
    plt.scatter(timeECG[mxs_indices], waveData[mxs_indices], color='r')
    plt.title("Onda " + nice[i] + "\n Frecuencia cardíaca (bpm) basada en picos R : " + str(mean_bpm) + "\n Frecuencia cardíaca (bpm) basada en los intervalos R-R :" + str(mean_bpm_from_rr)) 
    ### Se agrega titulo a los graficos con dos diferentes frecuencias cardiacas, dadas por los dos diferentes metodos

    plt.show()

    ### Frecuencia cardíaca por los dos diferentes metodos, se imprime tambien en pantalla (ya se imprime dentro del grafico)
    
    print("Onda " + nice[i] + "\n Frecuencia cardíaca (bpm) basada en picos R : " + str(mean_bpm) + "\n Frecuencia cardíaca (bpm) basada en los intervalos R-R :" + str(mean_bpm_from_rr))



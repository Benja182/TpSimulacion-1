import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, norm, expon, poisson, uniform, kstest


"""
matplotlib.pyplot es una biblioteca de gráficos 2D para Python
numpy es una biblioteca para el cálculo numérico en Python
scipy.stats es un módulo de SciPy que contiene funciones para trabajar con distribuciones estadísticas
"""

# Generador de números aleatorios - Metodo congruencial lineal

def generar_random(seed, a, c, m, n):
    """
    Genera una secuencia de números pseudoaleatorios utilizando el método congruencial lineal.
    
    Parámetros:
    seed (int): Semilla inicial para el generador.
    a (int): Multiplicador.
    c (int): Incremento.
    m (int): Módulo.
    n (int): Número de valores a generar.
    
    Retorna:
    list: Lista de números pseudoaleatorios generados.
    """
    randoms = []
    epsilon = 1e-10  # Para evitar problemas de redondeo

    for _ in range(n):
        seed = (a * seed + c) % m
        rnd = seed / m

        rnd = min(max(rnd, epsilon), 1-epsilon)
        randoms.append(rnd)

    return randoms

# Generador de distribuciones

#Uniforme
def generar_uniforme(randoms, A, B):
    """
    Genera números aleatorios uniformemente distribuidos en el intervalo [A, B].

    Parámetros:
    randoms (list): Lista de números pseudoaleatorios generados.
    A (float): Límite inferior del intervalo.
    B (float): Límite superior del intervalo.

    Retorna:
    list: Lista de números aleatorios uniformemente distribuidos.
    """
    return [round(A + (B - A) * rnd, 2) for rnd in randoms]
      
#Exponencial

def generar_exponencial(randoms, media):
    """
    Genera números aleatorios siguiendo una distribución exponencial.
    Utiliza la inversa de la función de distribución acumulativa (CDF) para generar los números.

    Parámetros:
    randoms (list): Lista de números pseudoaleatorios generados.
    media (float): Media de la distribucion.

    Retorna:
    list: Lista de números aleatorios exponencialmente distribuidos.
    """
    lambda_val = 1 / media
    epsilon = 1e-10  # Para evitar problemas de redondeo
    return [round(-np.log(1 - rnd + epsilon) / lambda_val, 2) for rnd in randoms]


#Normal

def generar_normal(randoms, mu, sigma):
    """
    Genera números aleatorios siguiendo una distribución normal.
    Sigue el metodo de Box-Muller.

    Parámetros:
    randoms (list): Lista de números pseudoaleatorios generados.
    mu (float): Media de la distribución normal.
    sigma (float): Desviación estándar de la distribución normal.

    Retorna:
    list: Lista de números aleatorios normalmente distribuidos.
    """

    n = len(randoms)//2
    normal_randoms = []
    for i in range(0, n, 2):
        u1 = randoms[i]
        u2 = randoms[i + 1]

        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        

        normal_randoms.append(round(mu + z0 * sigma, 2))

        if i + 1 < n:
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            normal_randoms.append(round(mu + z1 * sigma, 2))
    return normal_randoms


#Poisson

def generar_poisson(randoms, lambda_val):
    """
    Genera números aleatorios siguiendo una distribución de Poisson.

    Parámetros:
    randoms (list): Lista de números pseudoaleatorios generados.
    lambda_val (float): Tasa de ocurrencia media.

    Retorna:
    list: Lista de números aleatorios distribuidos según la distribución de Poisson.
    """
    e_neg_lambda = np.exp(-lambda_val)
    poisson_randoms = []

    index = 0

    for _ in range(len(randoms)//10):
        P = 1
        X = -1

        while P >= e_neg_lambda:
            rnd = randoms[index]
            P *= rnd
            X += 1
            index += 1

            if index >= len(randoms):
             index = 0
        
        poisson_randoms.append(X)

    return poisson_randoms

# Graficador de distribuciones

def graficar_histograma(data, title, xlabel, ylabel, bins=10, density=True):
    """
    Genera un histograma de los datos proporcionados.

    Parámetros:
    data (list): Lista de datos a graficar.
    title (str): Título del gráfico.
    xlabel (str): Etiqueta del eje x.
    ylabel (str): Etiqueta del eje y.
    bins (int): Número de bins para el histograma. Por defecto es 10.
    """
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='g')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()        

# Pruebas de bondad

#chi cuadrado
def chi_cuadrado(data, intervalos, tipo, parametros):
    """
    Evaluar la bondad de ajuste de los datos utilizando la prueba de chi-cuadrado.

    Parámetros:
    data (list): Lista de datos a evaluar.
    intervalos (int): Número de intervalos para el histograma.
    tipo (str): Tipo de distribución (uniforme, exponencial, normal o poisson).
    parametros (tuple): Parámetros de la distribución.
    
    """
    
    frec_observada, bordes = np.histogram(data, bins=intervalos)
    frec_esperada = []

    n_data = len(data)
    n_bordes = len(bordes) - 1

    if tipo == "uniforme":
        A, B = parametros
        distancia = B - A

    elif tipo == "exponencial":
        media = parametros[0]
    
    elif tipo == "normal":
        mu, sigma = parametros
    
    elif tipo == "poisson":
        lambda_valor = parametros[0]

    for i in range(n_bordes):
        borde_inferior = bordes[i]
        borde_superior = bordes[i + 1]

        if tipo == "poisson":
            prob = poisson.cdf(borde_superior, mu=lambda_valor) - poisson.cdf(borde_inferior-1, mu=lambda_valor) 
                  
        elif tipo == "exponencial":
            prob = expon.cdf(borde_superior, scale=media) - expon.cdf(borde_inferior, scale=media)
            
        elif tipo == "normal":
            prob = norm.cdf(borde_superior, loc=mu, scale=sigma) - norm.cdf(borde_inferior, loc=mu, scale=sigma)
            
        elif tipo == "uniforme":
            prob = uniform.cdf(borde_superior, A, distancia) - uniform.cdf(borde_inferior, A, distancia)
        
        frec_esperada.append(prob * n_data)
      
    frec_esp_total = sum(frec_esperada)
    frec_obs_total = sum(frec_observada)
    
    #Es necesario hacer un ajuste si las frecuencias acumuladas son distintas
    if frec_esp_total != frec_obs_total:
        ajuste = frec_obs_total / frec_esp_total
        frec_esperada = [frec * ajuste for frec in frec_esperada]
    
    #Esta funcion, incluida en el modulo scipy.stats permite realizar la prueba de bondad
    chi2, p_valor = chisquare(frec_observada, frec_esperada)
    

    """
    Alternativa para el ajuste, segun Deepseek, se debe realizar lo siguiente y NO el ajuste hecho:
    # Verificar y agrupar categorías con frecuencias esperadas < 5
    frec_obs = []
    frec_esp = []
    temp_obs = 0
    temp_esp = 0
    
    for obs, esp in zip(frec_observada, frec_esperada):
        if esp < 5:
            temp_obs += obs
            temp_esp += esp
        else:
            if temp_obs > 0:
                frec_obs.append(temp_obs)
                frec_esp.append(temp_esp)
                temp_obs = 0
                temp_esp = 0
            frec_obs.append(obs)
            frec_esp.append(esp)
    
    if temp_obs > 0:
        frec_obs.append(temp_obs)
        frec_esp.append(temp_esp)
  
    """



    return chi2, p_valor


#Kolmogorov-Smirnov

def ks_test(data, tipo, parametros):

    """
    Realiza la prueba de Kolmogorov-Smirnov para una distribución específica.
    
    Parámetros:
    - data: array con los datos muestrales
    - intervalos: cantidad de intervalos de la distribucion
    - tipo: tipo de distribución ('uniforme', 'normal' o 'exponencial')
    - parametros: parámetros de la distribución (None para estimarlos de los datos)

    
    Retorna:
    - ks_valor: estadístico D de K-S
    - p_valor: valor p de la prueba
    - result: decisión de la prueba (Rechazar/No rechazar H0)
    """
    if tipo == "uniforme":
        A, B = parametros
        distancia = B - A

    elif tipo == "exponencial":
        media = parametros[0]
    
    elif tipo == "normal":
        mu, sigma = parametros


    # Crear objeto de distribución correspondiente
    if tipo == 'uniforme':
        dist = uniform(loc= A, scale= B - A)
    elif tipo == 'normal':
        dist = norm(loc=mu, scale=sigma)
    elif tipo == 'exponencial':
        dist = expon(scale=media)
  
    
    # Realizar prueba K-S
    ks_valor, p_valor = kstest(data, dist.cdf)
    return ks_valor, p_valor
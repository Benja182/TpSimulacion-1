import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, expon, poisson, uniform, kstest, chi2
import pandas as pd
from collections import Counter
import os


"""
-matplotlib.pyplot es una biblioteca de gráficos 2D para Python
-numpy es una biblioteca para el cálculo numérico en Python
-scipy.stats es un módulo de SciPy que contiene funciones para trabajar con distribuciones estadísticas
-pandas es una biblioteca para la manipulación y análisis de datos (manejo de archivos, especialmente los excel y csv)
-collections.Counter es una clase de Python que permite contar elementos hashables (como listas o tuplas) y crear diccionarios de conteo.
-os es una biblioteca de Python que proporciona funciones para interactuar con el sistema operativo.
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
    for i in range(0, n-1, 2):
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

def graficar_histograma(data, title, xlabel, ylabel, bins=10, density=False):
    """
    Genera un histograma de los datos proporcionados.

    Parámetros:
    data (list): Lista de datos a graficar.
    title (str): Título del gráfico.
    xlabel (str): Etiqueta del eje x.
    ylabel (str): Etiqueta del eje y.
    bins (int): Número de bins para el histograma. Por defecto es 10.
    """
    plt.hist(data, bins=bins, density=density, alpha=0.6, color='g', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()        

# Pruebas de bondad

#chi cuadrado

def agrupar_clases(fo_vals, fe_vals, clases, umbral=5):
    """
    Agrupa las clases de acuerdo a un umbral dado.
    Esto se hace debido a que la prueba de Chi Cuadrado lo requiere

    Parametros:
    - fo_vals: lista de frecuencias observadas
    - fe_vals: lista de frecuencias esperadas
    - clases: lista de clases
    - umbral: valor mínimo para agrupar clases
    
    """


    fo_agrup, fe_agrup, clases_agrup = [], [], []
    temp_fo = temp_fe = 0
    temp_clase = []

    for i in range(len(fe_vals)):
        temp_fo += fo_vals[i]
        temp_fe += fe_vals[i]
        temp_clase.append(clases[i])

        if temp_fe >= umbral:
            clases_agrup.append(f"{temp_clase[0]} – {temp_clase[-1]}")
            fo_agrup.append(temp_fo)
            fe_agrup.append(temp_fe)
            temp_fo = temp_fe = 0
            temp_clase = []

    if temp_fe > 0:
        if len(clases_agrup) > 0:
            fo_agrup[-1] += temp_fo
            fe_agrup[-1] += temp_fe
            inicio = clases_agrup[-1].split(" – ")[0]
            clases_agrup[-1] = f"{inicio} – {temp_clase[-1]}"
        else:
            clases_agrup.append(f"{temp_clase[0]} – {temp_clase[-1]}")
            fo_agrup.append(temp_fo)
            fe_agrup.append(temp_fe)

    return clases_agrup, fo_agrup, fe_agrup


def calcular_chi2(fo, fe):
    """
    Calcula la estadística Chi-Cuadrado para las frecuencias observadas y esperadas.

    Parametros:
    - fo: lista de frecuencias observadas
    - fe: lista de frecuencias esperadas
    """

    return [((f_o - f_e) ** 2) / f_e if f_e > 0 else 0 for f_o, f_e in zip(fo, fe)]


def prueba_chi_cuadrado(valores, tipo, params, bins=10, alpha=0.05, archivo_salida="resultado_chi2.xlsx", gl=0):
    """
    Realiza la prueba de bondad de ajuste Chi-Cuadrado para una distribución específica.

    Parametros:
    - valores: array con los datos muestrales
    - tipo: tipo de distribución ('uniforme', 'exponencial', 'normal' o 'poisson')
    - params: parámetros de la distribución (None para estimarlos de los datos)
    - bins: número de intervalos para el histograma (solo para distribuciones continuas)
    - alpha: nivel de significancia para la prueba
    - archivo_salida: nombre del archivo de salida para guardar los resultados
    - gl: grados de libertad (opcional, se calcula automáticamente si no se proporciona)

    """


    n = len(valores)

    if tipo == "poisson":
        valores_redondeados = [int(round(v)) for v in valores if v >= 0]
        fo_counter = Counter(valores_redondeados)
        lamb = params[0] if params else np.mean(valores_redondeados)
        n = len(valores_redondeados)

        max_val = max(fo_counter.keys())
        clases = list(range(max_val + 1))
        fo_vals = [fo_counter.get(i, 0) for i in clases]
        fe_vals = [poisson.pmf(i, mu=lamb) * n for i in clases]

        clases_agrup, fo_agrup, fe_agrup = agrupar_clases(fo_vals, fe_vals, clases)

    else:
        fo_vals, bordes = np.histogram(valores, bins=bins)
        fe_vals = []
        clases = []

        for i in range(len(bordes) - 1):
            a, b = bordes[i], bordes[i + 1]

            if tipo == 'uniforme':
                A, B = params
                prob = uniform.cdf(b, loc=A, scale=B - A) - uniform.cdf(a, loc=A, scale=B - A)
            elif tipo == 'exponencial':
                media = params[0]
                prob = expon.cdf(b, scale=media) - expon.cdf(a, scale=media)
            elif tipo == 'normal':
                mu, sigma = params
                prob = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
            else:
                raise ValueError("Distribución no reconocida.")

            fe = prob * n
            fe_vals.append(fe)
            clases.append(f"[{a:.2f}, {b:.2f})")

        clases_agrup, fo_agrup, fe_agrup = agrupar_clases(fo_vals, fe_vals, clases)

    contribuciones = calcular_chi2(fo_agrup, fe_agrup)
    chi2_stat = sum(contribuciones)
    k = len(fo_agrup)
    m = gl  # Podés ajustar esto si estimás parámetros
    v = k - 1 - m
    chi2_critico = chi2.ppf(1 - alpha, df=v)
    decision = "\033[32mNO se rechaza H0\033[0m" if chi2_stat <= chi2_critico else "\033[31mSe rechaza H0\033[0m"
   
    resumen = {
        "Cantidad de datos": n,
        "Clases agrupadas (k)": k,
        "Parámetros estimados (m)": m,
        "Grados de libertad (v = k-1-m)": v,
        "Chi² calculado": round(chi2_stat, 4),
        f"Valor crítico (α={alpha})": round(chi2_critico, 4),
        "Resultado": decision
    }
    if tipo == "poisson":
        resumen["Lambda estimado"] = round(lamb, 4)

    df_tabla = pd.DataFrame({
        "Clase" if tipo == "poisson" else "Intervalo": clases_agrup,
        "fo": fo_agrup,
        "fe": [round(f, 2) for f in fe_agrup],
        "((fo-fe)^2)/fe": [round(c, 4) for c in contribuciones]
    })

    # Mostrar resultado en consola
    print("\n--- Resultado Chi-Cuadrado ---")
    for key, val in resumen.items():
        print(f"{key}: {val}")

    resumen ["Resultado"] = decision[5:-4]
    df_resumen = pd.DataFrame(list(resumen.items()), columns=["Descripción", "Valor"])

    
    archivo_salida = input("Ingrese el nombre del archivo de salida (por ejemplo: resultados_chi.xlsx): ").strip()
    exportar_resultados(df_tabla, df_resumen, archivo_salida)


# Manejo de archivos


def cargar_datos_desde_archivo(nombre_archivo):
    """
    Intenta leer el archivo y devuelve los datos si es válido.

    parametros:
    nombre_archivo (str): Nombre del archivo a leer.
    """
   
    try:
        extension = os.path.splitext(nombre_archivo)[1].lower()

        if extension == ".csv":
            df = pd.read_csv(nombre_archivo)
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(nombre_archivo)
        else:
            print("Formato no soportado. Usa un archivo .csv o .xlsx")
            return 

        if "Valor" not in df.columns:
            print("El archivo debe tener una columna llamada 'Valor'.")
            return 

        return df["Valor"]

    except FileNotFoundError:
        print(f"No se encontró el archivo: {nombre_archivo}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")



def generar_archivo_csv(nombre_archivo, valores):
    """
    Genera un archivo CSV con una lista de números aleatorios.

    parametros:
    nombre_archivo (str): Nombre del archivo a generar.
    valores (list): Lista de números aleatorios a guardar en el archivo.
    """
    try:
        extension = os.path.splitext(nombre_archivo)[1].lower()

        if not nombre_archivo.endswith(".csv"):
            nombre_archivo += ".csv"
        
        if os.path.exists(nombre_archivo):
            
            while True:
              respuesta = input(f"\033[33mEl archivo {nombre_archivo} ya existe. ¿Desea sobrescribirlo? (s/n): \033[0m").strip().lower()
              if respuesta in ['s', 'n']:
                  break
              else:
                  print("Respuesta no válida. Por favor, ingrese 's' o 'n'.")

            if respuesta != 's':
                print("Operación cancelada.")
                return

        df = pd.DataFrame(valores, columns=['Valor'])
        df.to_csv(nombre_archivo, index=False)

        print(f"Archivo {nombre_archivo} creado exitosamente")
        return True

    except PermissionError:
        print(f"No se tienen los permisos como para escribir en la direccion dada")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

    



def exportar_resultados(df_tabla, df_resumen, archivo_salida):
    """
    Este método exporta los resultados de la prueba Chi-Cuadrado a un archivo Excel.

    parametros:
    df_tabla (DataFrame): DataFrame con los resultados de la prueba Chi-Cuadrado.
    df_resumen (DataFrame): DataFrame con el resumen de la prueba.
    

    """

    if not archivo_salida.endswith(".xlsx"):
        archivo_salida += ".xlsx"

    if os.path.exists(archivo_salida):
        while True:
            respuesta = input(f"\033[33mEl archivo {archivo_salida} ya existe. ¿Desea sobrescribirlo? (s/n): \033[0m").strip().lower()
            if respuesta in ['s', 'n']:
                break
            else:
                print("Respuesta no válida. Por favor, ingrese 's' o 'n'.")

        if respuesta != 's':
            print("Operación cancelada.")
            return

    with pd.ExcelWriter(archivo_salida) as writer:
        df_tabla.to_excel(writer, sheet_name="Chi2 Detalle", index=False)
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
    print(f"\nResultado exportado a: {archivo_salida}")
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import chi2, norm, expon, poisson, uniform
from collections import Counter

"""
EL PROBLEMA EN ESTA VERSION:
- FALTA MAS TESTEOS.
- ACA CUANDO USO UNA MUESTRA DE 50.000 CON VALORES DISCRETOS, COMO LA MUESTRA GENERADA CON LA VARIABLE POISSON, ME GENERA BASICAMENTE
UNA NORMAL, PERO EL TEST DE CHI CUADRADO ME LA RECHAZA.
- GENERA VALORES DE CHI CHUADRADO MUY GRANDES, 
quiero entender poorque genera valores tan grandes en algunos casos como:
- muestra 50.000 poisson. prueba de bondad H0: normal. genera un chi cuadrado enorme


CORRECIONES:
- POISSON FUNCIONA CORRECTAMENTE PARA MUESTRAS CHICAS. EJ 50 DATOS
- PARA UNA MUESTRA DE 50.000 CHI CUADRADO RECHAZA POISSON. ESTA BIEN PORQUE TIENDE A LA NORMAL.
- CHI CUADRADO FUNCIONA BIEN - TESTIE VALORES DEL PDF pruebas de bondad de ajuste. 

QUE OPINO:
- creo que cuando la muestra es muy grande 50000, la prueba de chi cuadrado es muy sensible.

CUANDO FUNCIONA CORRECTAMENTE:
- tipo muestra: normal - tamaño: 50.000 - intervalos: 200 - H0: normal # funciona bien en casos similares
- tipo muestra: uniforme - tamaño: 50.000 - intervalos: 200 - H0: uniforme # funciona bien en casos similares
- tipo muestra: exponencial - tamaño: 50.000 - intervalos: 200 - H0: exponencial # funciona bien en casos similares
- tipo muestra: poisson - tamaño: 50 - intervalos: 8 - H0: poisson # funciona bien en casos similares

jaja como se complico integrar la prueba de chi cuadrado
aparecen conflictos cuando se usan valores discretos.

TESTEAR:
- muestras y pruebas de distinto tipo.

"""

# ------------------ Funciones Auxiliares ------------------

def cargar_datos_desde_archivo(nombre_archivo):
    """
    Intenta leer el archivo y devuelve los datos si es válido.
    """
    try:
        extension = os.path.splitext(nombre_archivo)[1].lower()

        if extension == ".csv":
            df = pd.read_csv(nombre_archivo)
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(nombre_archivo)
        else:
            print("Formato no soportado. Usa un archivo .csv o .xlsx")
            return None

        if "Valor" not in df.columns:
            print("El archivo debe tener una columna llamada 'Valor'.")
            return None

        return df["Valor"]

    except FileNotFoundError:
        print(f"No se encontró el archivo: {nombre_archivo}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

    return None


def graficar_histograma(datos, titulo, num_bins):
    """
    Genera el histograma.
    """
    plt.hist(datos, bins=num_bins, edgecolor="black", alpha=0.7, density=True)
    plt.title(titulo)
    plt.xlabel("Valor")
    plt.ylabel("Densidad de Frecuencia")
    plt.grid(True)
    plt.show()


def agrupar_clases(fo_vals, fe_vals, clases, umbral=5):
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
    return [((f_o - f_e) ** 2) / f_e if f_e > 0 else 0 for f_o, f_e in zip(fo, fe)]


def exportar_resultados(df_tabla, df_resumen, archivo_salida):
    if not archivo_salida.endswith(".xlsx"):
        archivo_salida += ".xlsx"
    with pd.ExcelWriter(archivo_salida) as writer:
        df_tabla.to_excel(writer, sheet_name="Chi2 Detalle", index=False)
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
    print(f"\nResultado exportado a: {archivo_salida}")



# ------------------ Función Principal ------------------

def prueba_chi_cuadrado(valores, tipo, params, bins=10, alpha=0.05, archivo_salida="resultado_chi2.xlsx"):
    n = len(valores)

    if tipo == "poisson":
        valores_redondeados = [int(round(v)) for v in valores if v >= 0]
        fo_counter = Counter(valores_redondeados)
        lamb = np.mean(valores_redondeados)
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
    m = 0  # Podés ajustar esto si estimás parámetros
    v = k - 1 - m
    chi2_critico = chi2.ppf(1 - alpha, df=v)
    decision = "NO se rechaza H0" if chi2_stat <= chi2_critico else "Se rechaza H0"

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

    df_resumen = pd.DataFrame(list(resumen.items()), columns=["Descripción", "Valor"])

    # Mostrar resultado en consola
    print("\n--- Resultado Chi-Cuadrado ---")
    for key, val in resumen.items():
        print(f"{key}: {val}")

    archivo_salida = input("Ingrese el nombre del archivo de salida (por ejemplo: resultados_chi.xlsx): ").strip()
    exportar_resultados(df_tabla, df_resumen, archivo_salida)


if __name__ == "__main__":
    # Validar archivo y cargar datos
    datos = None
    while datos is None:
        archivo = input("Ingrese el nombre del archivo (.csv o .xlsx): ")
        datos = cargar_datos_desde_archivo(archivo)

    # Validar número de intervalos
    while True:
        try:
            bins = int(input("Ingrese la cantidad de intervalos (entre 3 y 230): "))
            if 3 <= bins <= 230:
                break
            else:
                print("El número debe estar entre 3 y 230.")
        except ValueError:
            print("Entrada inválida. Debe ser un número entero.")

    # Validar nivel de significancia
    while True:
        try:
            alpha = float(input("Ingrese el nivel de significancia (por ejemplo 0.05): "))
            if 0.01 <= alpha <= 0.1:
                break
            else:
                print("Debe ser un valor entre 0.01 y 0.1.")
        except ValueError:
            print("Entrada inválida. Debe ser un número decimal.")
    
    # Graficar una vez con todos los datos validados
    graficar_histograma(datos, f"Histograma de {archivo}", bins)
    
    
    # Preguntar tipo de distribución
    tipo = None
    while tipo not in ["uniforme", "normal", "exponencial", "poisson"]:
        tipo = input("Ingrese el tipo de distribución ('uniforme', 'normal', 'exponencial', 'poisson'): ").strip().lower()

    # Calcular parámetros desde los datos
    if tipo == "uniforme":
        A = min(datos)
        B = max(datos)
        params = (A, B)
        print(f"\n[Estimación] Uniforme: A = {A:.4f}, B = {B:.4f}")
    elif tipo == "normal":
        mu = np.mean(datos)
        sigma = np.std(datos, ddof=1)
        params = (mu, sigma)
        print(f"\n[Estimación] Normal: μ = {mu:.4f}, σ = {sigma:.4f}")
    elif tipo == "exponencial":
        media = np.mean(datos)
        params = (media,)
        print(f"\n[Estimación] Exponencial: media = {media:.4f}")
    elif tipo == "poisson":
        lamb = np.mean(datos)
        params = (lamb,)
        print(f"\n[Estimación] Poisson: λ = {lamb:.4f}")

    # Graficar
    graficar_histograma(datos, f"Histograma de {archivo}", bins)

    # prueba_chi_cuadrado
    prueba_chi_cuadrado(datos, tipo, params, bins, alpha)


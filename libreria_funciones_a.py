# imports
import matplotlib.pyplot as plt 
import numpy as np 
import csv

# funciones
def metodo_congruencial_lineal(seed, a, c, m, n):
    """
    Genera una secuencia de números pseudoaleatorios usando el método congruencial lineal.
    :param seed: Valor inicial o semilla
    :param a: Multiplicador
    :param c: Incremento
    :param m: Módulo
    :param n: Cantidad de números a generar
    :return: Lista de números pseudoaleatorios
    """
    numeros = []
    X = seed
    epsilon = 1e-10
    
    for _ in range(n):
        X = (a * X + c) % m
        rnd = X / m
        
        # Aseguramos que esté en el rango (0,1) evitando los bordes peligrosos
        rnd = min(max(rnd, epsilon), 1 - epsilon)  # Mantenerlo dentro del rango seguro
        numeros.append(rnd)
        
    return numeros


def generar_uniforme_mcl(seed, a, c, m, n, A, B):
    """
    Genera números con distribución uniforme en [A, B] usando el método congruencial lineal.
    :param A: Límite inferior
    :param B: Límite superior
    """
    uniformes_01 = metodo_congruencial_lineal(seed, a, c, m, n)
    distribucion_uniforme = [round(A + u * (B - A),4) for u in uniformes_01]
    return distribucion_uniforme


def generar_exponencial_mcl(seed, a, c, m, n, media):
    """
    Genera números con distribución exponencial negativa usando el método congruencial lineal.
    :param media: Media de la distribución (u)
    """
    lambda_val = 1 / media  # λ = 1 / u
    uniformes = metodo_congruencial_lineal(seed, a, c, m, n)
    
    # Evitar log(0) asegurándonos de que (1 - RND) no sea 0
    epsilon = 1e-10  
    exponenciales = [-np.log(max(1 - u, epsilon)) / lambda_val for u in uniformes]
    
    return exponenciales


def generar_normal_mcl(seed, a, c, m, n, media, desviacion):
    """
    Genera números con distribución normal usando el método de Box-Muller y el método congruencial lineal.
    :param media: Media de la distribución
    :param desviacion: Desviación estándar
    """
    # Generar 2*n números uniformes
    uniformes = metodo_congruencial_lineal(seed, a, c, m, 2 * n)

    normales = []
    i = 0
    while len(normales) < n:
        U1, U2 = uniformes[i], uniformes[i + 1]
        R = np.sqrt(-2 * np.log(U1))
        Theta = 2 * np.pi * U2

        Z0 = R * np.cos(Theta)
        Z1 = R * np.sin(Theta)

        normales.append(media + Z0 * desviacion)
        if len(normales) < n:
            normales.append(media + Z1 * desviacion)

        i += 2
    
    return normales  
    


def generar_poisson_mcl(seed, a, c, m, n, lambda_val):
    """
    Genera números con distribución de Poisson usando el método de multiplicación.
    :param lambda_val: Parámetro lambda de la distribución (tasa de eventos esperados)
    """
    valores_poisson = []
    uniformes = metodo_congruencial_lineal(seed, a, c, m, n) 
    e_neg_lambda = np.exp(-lambda_val)  # e^(-lambda)

    index = 0
    for _ in range(n):
        P = 1
        X = -1
        while P >= e_neg_lambda:
            U = uniformes[index]  # Tomar un número aleatorio
            P *= U
            X += 1
            index += 1
            if index >= len(uniformes):  # Reabastecer si es necesario
                uniformes = metodo_congruencial_lineal(seed, a, c, m, n * 10)
                index = 0
        
        valores_poisson.append(X)

    return valores_poisson


def menu():
    print("\n=== MENÚ DE DISTRIBUCIONES ===")
    print("1. Distribución Uniforme")
    print("2. Distribución Exponencial")
    print("3. Distribución Normal")
    print("4. Distribución Poisson")
    print("5. Salir")


def ejecutar_uniforme():
    # UNIFORME
    print("---------------------- Distribucion uniforme ----------------------")

    # Pedir límites A y B al usuario
    A = float(input("Ingrese el límite inferior A de la distribución uniforme: "))
    B = float(input("Ingrese el límite superior B de la distribución uniforme: "))

    # Generar valores
    valores_uniformes = generar_uniforme_mcl(seed, a, c, m, cantidad, A, B)

    # Graficar histograma
    plt.hist(valores_uniformes, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Distribución Uniforme en [{A}, {B}]")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    guardar = input("¿Desea guardar los datos en un archivo CSV? (s/n): ").lower()
    if guardar == 's':
        nombre = input("Ingrese el nombre del archivo (sin extensión): ")
        guardar_en_csv(f"{nombre}.csv", valores_uniformes)


def ejecutar_exponencial():
    # EXPONENCIAL
    print("---------------------- Distribucion exponencial ----------------------")
    
    # Pedir media al usuario
    media = float(input("Ingrese la media de la distribución exponencial: "))
    
    # Generar valores
    valores_exponenciales = generar_exponencial_mcl(seed, a, c, m, cantidad, media)

    # Graficar histograma
    plt.hist(valores_exponenciales, bins=30, edgecolor='black', alpha=0.7, density=True)
    plt.title(f"Distribución Exponencial Negativa (Media={media})")
    plt.xlabel("Valor")
    plt.ylabel("Densidad de Frecuencia")
    plt.grid(True)
    plt.show()
    
    guardar = input("¿Desea guardar los datos en un archivo CSV? (s/n): ").lower()
    if guardar == 's':
        nombre = input("Ingrese el nombre del archivo (sin extensión): ")
        guardar_en_csv(f"{nombre}.csv", valores_exponenciales)


def ejecutar_normal():
    # NORMAL
    print("---------------------- Distribucion normal ----------------------")
    
    # Pedir media y desviación estándar al usuario
    media = float(input("Ingrese la media de la distribución normal: "))
    desviacion = float(input("Ingrese la desviación estándar de la distribución normal: "))
    
    # Generar valores
    valores_normales = generar_normal_mcl(seed, a, c, m, cantidad, media, desviacion)

    # Graficar histograma
    plt.hist(valores_normales, bins=30, edgecolor='black', alpha=0.7, density=True)
    plt.title(f"Distribución Normal (Media={media}, Desviación={desviacion})")
    plt.xlabel("Valor")
    plt.ylabel("Densidad de Frecuencia")
    plt.grid(True)
    plt.show()

    guardar = input("¿Desea guardar los datos en un archivo CSV? (s/n): ").lower()
    if guardar == 's':
        nombre = input("Ingrese el nombre del archivo (sin extensión): ")
        guardar_en_csv(f"{nombre}.csv", valores_normales)


def ejecutar_poisson():
    # POISSON
    print("---------------------- Distribucion Poisson ----------------------")
    
    # Solicitar parámetro de la distribución de Poisson
    lambda_val = float(input("Ingrese el valor de lambda para la distribución de Poisson: "))
    
    # Generar valores
    valores_poisson = generar_poisson_mcl(seed, a, c, m, cantidad, lambda_val)

    # Graficar histograma
    plt.hist(valores_poisson, bins=range(min(valores_poisson), max(valores_poisson) + 1), edgecolor='black', alpha=0.7, density=True)
    plt.title(f"Distribución de Poisson (λ={lambda_val})")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia relativa")
    plt.xticks(range(min(valores_poisson), max(valores_poisson) + 1))
    plt.grid(axis='y')
    plt.show()

    guardar = input("¿Desea guardar los datos en un archivo CSV? (s/n): ").lower()
    if guardar == 's':
        nombre = input("Ingrese el nombre del archivo (sin extensión): ")
        guardar_en_csv(f"{nombre}.csv", valores_poisson)


def guardar_en_csv(nombre_archivo, datos):
    """
    Guarda una lista de datos en un archivo CSV.
    """
    with open(nombre_archivo, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Valor"])
        for valor in datos:
            writer.writerow([valor])
    print(f"Datos guardados en '{nombre_archivo}' exitosamente.")


if __name__ == "__main__":
    # Parámetros del generador MCL
    seed = 7
    a = 1103515245
    c = 12345
    m = 2**31

    # Validar cantidad de numeros a generar
    while True:
        try:
            cantidad = int(input("Ingrese la cantidad de números a generar: "))
            if 0 < cantidad <= 50000:
                break
            else:
                print("El número debe estar entre 1 y 50000.")
        except ValueError:
            print("Entrada no válida. Intente de nuevo.")
            
    # Menú interactivo
    while True:
        menu()
        opcion = input("Seleccione una opción: ")
        if opcion == '1':
            ejecutar_uniforme()
        elif opcion == '2':
            ejecutar_exponencial()
        elif opcion == '3':
            ejecutar_normal()
        elif opcion == '4':
            ejecutar_poisson()
        elif opcion == '5':
            print("Saliendo del programa. ¡Hasta la próxima!")
            break
        else:
            print("Opción no válida. Intente de nuevo.")










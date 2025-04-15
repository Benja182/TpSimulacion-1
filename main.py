import simToolbox as sim
import statistics

# Funciones auxiliares

def menu(opciones):
    for i in range(len(opciones)):
        print(f"{i+1}. {opciones[i]}")
    print("0. Salir")


def validar_entrada(mensaje, min_val=None, max_val=None, tipo="int"):
    
    """
    Valida la entrada del usuario asegurando que sea un número dentro de un rango específico.
    
    Parámetros:
    mensaje (str): Mensaje a mostrar al usuario para solicitar la entrada.
    min_val (float): Valor mínimo permitido. Por defecto es None.
    max_val (float): Valor máximo permitido. Por defecto es None.
    tipo (str): Tipo de dato esperado ('int' o 'float'). Por defecto es 'int'.
    """

    while True:
        try:
            if tipo == "int":
                entrada = int(input(mensaje))
            else:
                entrada = float(input(mensaje))

            if min_val is None:
                if entrada > max_val:
                    print(f"Por favor, ingrese un número menor o igual a {max_val}.")
                else:
                    break

            elif max_val is None:
                if entrada < min_val:
                    print(f"Por favor, ingrese un número mayor o igual a {min_val}.")
                else:
                    break

            elif min_val <= entrada <= max_val:
                break
            else:
                print(f"Por favor, ingrese un número entre {min_val} y {max_val}.")
        except ValueError:
            print("Entrada no válida. Intente de nuevo.")

    return entrada


# Distribuciones

def uniforme(seed, a, c, m):
    n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)
    A = validar_entrada("Ingrese el límite inferior del intervalo: ", 0, tipo="float")
    B = validar_entrada("Ingrese el límite superior del intervalo: ", A+0.000001, tipo="float")

    datos = sim.generar_uniforme(sim.generar_random(seed, a, c, m, n), A, B)
    intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)

    sim.graficar_histograma(datos, "Distribución Uniforme", "Frecuencia", "Valor", intervalos)
    print("\n ----------- \n")

    alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
    sim.prueba_chi_cuadrado(datos, "uniforme", [A, B], bins=intervalos, alpha=alpha)

    print("\n ----------- \n")
    return datos


def exponencial(seed, a, c, m):
    n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)
    media = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")

    datos = sim.generar_exponencial(sim.generar_random(seed, a, c, m, n), media)
    intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)

    sim.graficar_histograma(datos, "Distribución Exponencial", "Densidad", "Valor", intervalos)
    print("\n ----------- \n")

    alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
    sim.prueba_chi_cuadrado(datos, "exponencial", [media], bins=intervalos, alpha=alpha)

    print("\n ----------- \n")
    return datos


def normal(seed, a, c, m):
    n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)
    mu = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")
    sigma = validar_entrada("Ingrese la desviación estándar de la distribución: ", 0.00001, tipo="float")

    datos = sim.generar_normal(sim.generar_random(seed, a, c, m, n * 2), mu, sigma)
    intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)

    sim.graficar_histograma(datos, "Distribución Normal", "Densidad", "Valor", intervalos)
    print("\n ----------- \n")

    alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
    sim.prueba_chi_cuadrado(datos, "normal", [mu, sigma], bins=intervalos, alpha=alpha)

    print("\n ----------- \n")
    return datos


def poisson(seed, a, c, m):
    n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)
    media = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")

    datos = sim.generar_poisson(sim.generar_random(seed, a, c, m, n * 10), media)
    intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)

    sim.graficar_histograma(datos, "Distribución Poisson", "Frecuencia", "Valor", intervalos)
    print("\n ----------- \n")

    alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
    sim.prueba_chi_cuadrado(datos, "poisson", [media], bins=intervalos, alpha=alpha)

    print("\n ----------- \n")
    return datos


# Menú

def main():
    seed = 7
    a = 1103515245
    c = 12345
    m = 2**31

    opciones = [
        "Distribución Uniforme",
        "Distribución Exponencial",
        "Distribución Normal",
        "Distribución Poisson",
    ]

    print("Bienvenido a la herramienta de simulación de distribuciones")
    opcion = 1

    while opcion != 0:
        menu(opciones)
        opcion = int(input("Seleccione una opción: "))
        print("\n ----------- \n")

        print("¿Desea ingresar los datos desde un archivo?")
        print("1. Sí")
        print("2. No")
        ingreso = validar_entrada("Seleccione una opción: ", 1, 2, "int")

        print("\n ----------- \n")

        if opcion == 0:
            print("Saliendo...")
            break

        elif opcion == 1:
            # Distribucion Uniforme
            if ingreso == 1:
                # Cargar datos desde un archivo
                datos = None
                while datos is None:
                    nombre = input("Ingrese el nombre del archivo: ")
                    datos = sim.cargar_datos_desde_archivo(nombre)

                A = min(datos)
                B = max(datos)
                print(f"Límite inferior estimado: {A}")
                print(f"Límite superior estimado: {B}")

                intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)
                sim.graficar_histograma(datos, "Distribución Uniforme", "Frecuencia", "Valor", intervalos)

                print("\n ----------- \n")
                alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
                sim.prueba_chi_cuadrado(datos, "uniforme", [A, B], bins=intervalos, alpha=alpha, gl=2)
                print("\n ----------- \n")
            else:
                # Generar datos aleatorios
                datos = uniforme(seed, a, c, m)

            sim.generar_archivo_csv("uniforme", datos)

        elif opcion == 2:
            # Distribucion Exponencial
            if ingreso == 1:
                # Cargar datos desde un archivo
                datos = None
                while datos is None:
                    nombre = input("Ingrese el nombre del archivo: ")
                    datos = sim.cargar_datos_desde_archivo(nombre)

                media = statistics.mean(datos)
                print(f"Media estimada: {media}")

                intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)
                sim.graficar_histograma(datos, "Distribución Exponencial", "Frecuencia", "Valor", intervalos)

                print("\n ----------- \n")
                alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
                sim.prueba_chi_cuadrado(datos, "exponencial", [media], bins=intervalos, alpha=alpha, gl=1)
                print("\n ----------- \n")
            else:
                # Generar datos aleatorios
                datos = exponencial(seed, a, c, m)

            sim.generar_archivo_csv("exponencial", datos)

        elif opcion == 3:
            # Distribucion Normal
            if ingreso == 1:
                # Cargar datos desde un archivo
                datos = None
                while datos is None:
                    nombre = input("Ingrese el nombre del archivo: ")
                    datos = sim.cargar_datos_desde_archivo(nombre)

                mu = statistics.mean(datos)
                sigma = statistics.stdev(datos)
                print(f"Media estimada: {mu}")
                print(f"Desviación estándar estimada: {sigma}")

                intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)
                sim.graficar_histograma(datos, "Distribución Normal", "Densidad", "Valor", intervalos)

                print("\n ----------- \n")
                alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
                sim.prueba_chi_cuadrado(datos, "normal", [mu, sigma], bins=intervalos, alpha=alpha, gl=2)
                print("\n ----------- \n")
            else:
                # Generar datos aleatorios
                datos = normal(seed, a, c, m)

            sim.generar_archivo_csv("normal", datos)

        elif opcion == 4:
            # Distribucion Poisson
            if ingreso == 1:
                # Cargar datos desde un archivo
                datos = None
                while datos is None:
                    nombre = input("Ingrese el nombre del archivo: ")
                    datos = sim.cargar_datos_desde_archivo(nombre)

                media = statistics.mean(datos)
                print(f"Media estimada (λ): {media}")

                intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1, 25)
                sim.graficar_histograma(datos, "Distribución Poisson", "Frecuencia", "Valor", intervalos)

                print("\n ----------- \n")
                alpha = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001, 0.99999, "float")
                sim.prueba_chi_cuadrado(datos, "poisson", [media], bins=intervalos, alpha=alpha, gl=1)
                print("\n ----------- \n")
            else:
                # Generar datos aleatorios
                datos = poisson(seed, a, c, m)

            sim.generar_archivo_csv("poisson", datos)

        else:
            print("Opción no válida. Intente de nuevo.")

        input("Presionar enter para continuar")
        print("\n ----------- \n")


if __name__ == "__main__":
    main()

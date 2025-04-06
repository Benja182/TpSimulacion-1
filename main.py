import simToolbox as sim



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

def prueba_bondad(data, intervalos, tipo, parametros ):
    alfa = validar_entrada("Ingrese un nivel de significancia para valuar la prueba de bondad: ", 0.0000001,0.99999, "float")
    chi2, p_valor = sim.chi_cuadrado(data, intervalos, tipo, parametros)

    print(5*"\n")

    print("Los resultados de la prueba de bondad son:")
    print(f"Chi² = {chi2}")
    print(f"p-valor = {p_valor}")
    if p_valor <= alfa:
        print("❌ Se rechaza la hipótesis nula: los datos NO siguen la distribución esperada.")
    else:
        print("✅ No se puede rechazar la hipótesis nula: los datos podrían seguir la distribución.")

        if tipo != "poisson":
            print(2*"\n")
            print("¿Desea verificar la distribucion con una prueba de Kolmogorov-Smirnov?")
            
            while True:
                respuesta = (input("Ingresar SI o NO: ")).lower()
                if respuesta == "si" or respuesta == "no":
                    break
                else:
                    print("No ha ingresado una respuesta valida...")

            if respuesta == "NO": return

            print(5*"\n")
            
            ks_valor, p_valor = sim.ks_test(data, tipo, parametros)
            print("Los resultados de la prueba de bondad son:")
            print(f"KS = {ks_valor}")
            print(f"p-valor = {p_valor}")

            if p_valor <= alfa:
              print("❌ Se rechaza la hipótesis nula: los datos NO siguen la distribución esperada.")
            else:
              print("✅ No se puede rechazar la hipótesis nula: los datos podrían seguir la distribución.")




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

        print(5*"\n")
        
        if opcion == 0:
            print("Saliendo...")
            break

        elif opcion == 1:
            #Distribución Uniforme
            n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)

            A = validar_entrada("Ingrese el límite inferior del intervalo: ", 0)
            B = validar_entrada("Ingrese el límite superior del intervalo: ", A)

            datos = sim.generar_uniforme(sim.generar_random(seed, a, c, m, n), A, B)

            intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1)

            sim.graficar_histograma(datos, "Distribución Uniforme", "Frecuencia", "Valor", intervalos, True)

            print(5*"\n")
            prueba_bondad(datos, intervalos, "uniforme", [A,B])



        elif opcion == 2:
            #Distribución Exponencial
            n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)

            media = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")

            datos = sim.generar_exponencial(sim.generar_random(seed, a, c, m, n), media)

            intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1)

            sim.graficar_histograma(datos, "Distribución Exponencial", "Densidad", "Valor", intervalos, True)

            print(5*"\n")
            prueba_bondad(datos, intervalos, "exponencial", [media])
        


        elif opcion == 3:
            #Distribución Normal
            n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)

            mu = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")
            sigma = validar_entrada("Ingrese la desviación estándar de la distribución: ", 0.00001, tipo="float")

            datos = sim.generar_normal(sim.generar_random(seed, a, c, m, n*2), mu, sigma)

            intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1)

            sim.graficar_histograma(datos, "Distribución Normal", "Densidad", "Valor", intervalos, True)

            print(5*"\n")
            prueba_bondad(datos, intervalos, "normal", [mu, sigma])
        


        elif opcion == 4:
            #Distribución Poisson
            n = validar_entrada("Ingrese la cantidad de números a generar: ", 1, 50000)

            media = validar_entrada("Ingrese la media de la distribución: ", 0.00001, tipo="float")

            datos = sim.generar_poisson(sim.generar_random(seed, a, c, m, n*10), media)

            intervalos = validar_entrada("Ingrese la cantidad de intervalos para el histograma: ", 1)

            sim.graficar_histograma(datos, "Distribución Poisson", "Frecuencia", "Valor", intervalos, False)

            print(5*"\n")
            prueba_bondad(datos, intervalos, "poisson", [media])
        


        else:
            print("Opción no válida. Intente de nuevo.")

        input("Presionar enter para continuar")
        print(5*"\n")
          
            
            



if __name__ == "__main__":
    main()
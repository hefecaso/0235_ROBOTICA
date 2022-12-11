import cv2 #importamos la libreria de opencv
import numpy as np

def menu(): #definimos el menu de opciones
    print('\n        #############################')
    print('        #    Transformada de houg    #')
    print('        # ingrese "1" para circulos  #')
    print('        # ingrese "2" para lineas    #')
    print('        # ingrese "3" para salir     #')
    print('        ############################# ')

while True: #se establece el ciclo infinito
    menu() # llamamos a la funcion menù
    opcion = input("ingrese una opcion") #solicitamos al usuaro digitar una opcion


    if opcion == '1': #primera opcion
        img = cv2.imread('imagen5.jpeg') #creamos una variable en la cual se lee la imagen por medio de cv2
        src = cv2.medianBlur(img, 5) #se saca un filtro con una mediana de 5 con el fin de obtener la imagen mas fiel
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # se procesa la imagen por un filtro para escala de grises
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20, #se utiliza el gradiente de houg que permite analizar el borde de la imagen en blanco y negro para determinar los bordes circulares
                                    param1=50, param2=30, minRadius=1, maxRadius=50) # se establece los parametros de radio minimo y maximo para obtener en la imagen los circulos que deseamos
        circles = np.uint16(np.around(circles)) # se utiliza una arreglo de 16 bits para aproximar la cantidad de cituclos
        for i in circles[0,:]: # se establece la funcion encargada de contar los circulos
            # Dibuja la circusnferencia del círculo
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2) #se procede a identificar los circulos en la imagen
            # dibuja el centro del círculo
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3) #se procede a obtener el origen de los circulos
        cv2.imshow('Detector de Lineas', img) #se apertura un visualizador de open cv para mostrar las modificaciones en la imagen original
        cv2.waitKey()

    elif opcion == '2': #opcion para ver las lineas de una imagen
        img = cv2.imread('imagen4.jpeg') #aperturamos la lectura de la imaggen con opencv
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convertimos la imagen en escala de grises
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)#el comando canny permite detectar bordes el cual se le asigna dichos parametros para obtener los bordes reales o los mas aproximados
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10) #se utiliza del modulo de cv el comando hough lines utilizado para detectar las lineas parametrizando la imagen
        # con las esquinas, un angulo de 180 grados para las lineas de borde y una longitud minima de 100 y tomando como referencia que cada conjunto de 10 lineas agrupadas corresponde a una sola
        for line in lines:
            x1, y1, x2, y2 = line[0] #el conjunto de vectores nos mostrara cada linea
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
        #cv2.imshow('Bordes de Iamgen', edges)
        cv2.imshow('Detector de Lineas', img) #mostramos la imagen con las modificaciones de cv
        cv2.waitKey()


    elif opcion == '3':
        print("Saliendo del programa.")
        break

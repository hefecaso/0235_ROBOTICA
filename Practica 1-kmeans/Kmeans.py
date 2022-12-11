# ubicacion de imagen 'C:/Users/santo/Documents/GitHub/0235_ROBOTICA/Practica 1/input2.jpg
#ubicacion de imagen segmentada 'C:/Users/santo/Documents/GitHub/0235_ROBOTICA/Practica 1/out2.jpg'
"""
Practica 1 K-means 
Comunicaciones 2
"""
#Segmentación de imágenes simple utilizando el algoritmo de agrupamiento K-Means
#color clustering

import numpy as np #mportamos numpy para realizar operaciones matematicas
import cv2  #importamos opencv
import matplotlib.pyplot as plt #importamos para poder realizar graficas de precision


original_image = cv2.imread("C:/Users/santo/Documents/GitHub/0235_ROBOTICA/Practica 1-kmeans/imagen1.jpg") # abrimos la imagen a trabajar


img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB) # Conversión de BGR Colors Space a HSV
vectorized = img.reshape((-1,3)) #A continuación, convierte la imagen MxNx3 en una matriz Kx3 donde K=MxN y cada fila ahora es un vector en el espacio tridimensional de RGB.


vectorized = np.float32(vectorized)# Convertimos los valores unit8 en flotantes, ya que es un requisito del método k-means de OpenCV.

# Aquí estamos aplicando el agrupamiento k-means para que los píxeles alrededor de un color sean consistentes y proporcionen los mismos valores BGR/HSV
#definir criterios, número de grupos (K) y aplicar kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Defina criterios, número de conglomerados (K) y aplique k-means()

K = 10  # Vamos a agrupar con k = n, porque la imagen tendrá solo n colores.
attempts=10 # realizaremos 10 intentos andera para especificar el número de veces que se ejecuta el algoritmo usando diferentes etiquetas iniciales. El algoritmo devuelve las etiquetas que producen la mejor compacidad. Esta compacidad se devuelve como salida.

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS) #aplicamos kmeans con los criterios preeestablecidos

center = np.uint8(center) # Ahora convierte de nuevo a uint8
res = center[label.flatten()] #ahora tenemos que acceder a las etiquetas para regenerar la imagen agrupada
res2 = res.reshape((img.shape))  #res2 es el resultado del marco que ha sufrido un agrupamiento de k-means


#Ahora visualicemos el resultado de salida con K=n
figure_size = 15 #definimos el tamaño de la figura
plt.figure(figsize=(figure_size,figure_size)) #dibuja la figura con las dimensiones preestablecidas
plt.subplot(1,2,1),plt.imshow(img) #muestra la figura 
plt.title('Original'), plt.xticks([]), plt.yticks([]) #muestra la figura original
plt.subplot(1,2,2),plt.imshow(res2) #muestra la figura que sufrio la segmentacion
plt.title('K = %i' % K), plt.xticks([]), plt.yticks([]) #imprime el numero de clusters
plt.savefig("C:/Users/santo/Documents/GitHub/0235_ROBOTICA/Practica 1-kmeans/Segmentada_1.jpg")
plt.show() #finalmente muestra en una ventada flotante

#detector de bordes de canny
edges = cv2.Canny(img,100,200) #definimos una variable cuyos parametros son la imagen y los valores minimo y maximo 
plt.figure(figsize=(figure_size,figure_size)) #plot la figura con las dimensiones de la imagen
plt.subplot(1,2,1),plt.imshow(img) #dibuja la imagen
plt.title('Original Image'), plt.xticks([]), plt.yticks([]) #muestra la imagen originada
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray') 
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.savefig("C:/Users/santo/Documents/GitHub/0235_ROBOTICA/Practica 1-kmeans/bordes_1.jpg")
plt.show()
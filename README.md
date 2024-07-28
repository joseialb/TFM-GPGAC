# Clustering basado en la modificación de Grafos mediante Algoritmos Genéticos
## Autor: Jose Ignacio Alba Rodríguez

En este repositorio se presenta el código y los datasets empleados para la realización de este Trabajo de Fin de Master. La memoria del proyecto puede encontrarse en formato PDF.

El objetivo de este trabajo consiste en definir un nuevo algoritmo de clustering, llamado GPGAC (Graph Prunning based Genetic Algorithm for Clustering), que combina las nociones de centralidad en grafo y los algoritmos genéticos.
En GPGAC se parte de un grafo inicial G0 dado por los k0 vecinos más cercanos. Posteriormente, cada cromosoma de la población codifica una secuencia de modificaciones basados en los movimientos, definidos en el archivo "Movimientos.py".
Para obtener el valor de la función objetivo de cada uno de los cromosomas, se aplican cada uno de estos movimientos codificados iterativamente sobre el grafo, modificandolo en el proceso. Posteriormente, se extrae la clasificación dada por las componentes conexas del grafo y el valor de dicho cromosoma viene dado por una medida de calidad para esta clasificacion, por defecto el índice de Calinsky-Harabasz. El algoritmo GPGAC se encarga de recombinar y mutar los cromosomas como un algoritmo genético para tratar de extraer la mejor clasificación.

Los archivos que se encuentran en este repositorio son
* /Datasets  -> Carpeta que contiene todos los datasets empleados para la comparación del algoritmo
* GPGAC.py  -> Algoritmo Genético que encuentra la secuencia de modificaciones sobre el grafo que da lugar a la mejor clasificacion
* Movimientos.py  -> Cada uno de los movimientos considerados que pueden aparecer como cromosoma. Cada uno tiene una forma de mutar y de inicializar diferente
* main.py  -> Código empleado para clasificar cada uno de los datasets y representar los bidimensionales. Sirve a modo de ejemplo de uso
* TFM Jose Ignacio Alba.pdf  -> Memoria final del trabajo

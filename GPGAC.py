import random, time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Movimientos import *
from scipy.spatial.distance import pdist, squareform 
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, calinski_harabasz_score
from scipy.spatial import KDTree


class GPGAC():
    def __init__(self, datos, k, n0 = 50, max_it = 25, probMut=0.1, n_torneo = 3, n_elitismo = 2, metrica = 'calinski_harabasz_score', target = None, M = None, **kargs):
        self.datos = datos
        self.target = target
        self.metrica = metrica
        self.k0 = k 
        # Función objetivo
        if metrica == 'davies_bouldin_score':
            self.valor = float('inf')
            self.maximizar = False
        else:
            self.valor = float('-inf')
            self.maximizar = True
        # Matriz de distancias                
        if M is None: self.M = squareform(pdist(datos))
        else: self.M = M      
        self.M = self.M / np.max(self.M)                # Normalizamos las distancias
        # Matriz de orden de los vecinos. Se puede implementar con KDTrees para una mayor eficiencia en la inicializacion cuando k0 <<< len(datos)
        self.vecinos = np.argsort(self.M, axis = 1)     # self.vecinos[u,j] = v implica que v es el j-esimo vecino mas cercano de u
        self.reverso = np.zeros(self.vecinos.shape)     # self.reverso[u,v] = j implica que v es el j-esimo vecino mas cercano de u
        for i in range(len(datos)):
            for j in range(len(datos)):
                self.reverso[i, self.vecinos[i,j]] = j
        # Inicialización del grafo inicial
        aristas = [(i,v, {'weight': self.M[i,v]}) for i in self.vecinos[:,0] for v in self.vecinos[i, 1:self.k0 +1]  ]
        self.G0 = nx.Graph(aristas)      
        self.clas0 = self.clasificar()
        self.valor0, _ = self.evaluar(self.G0, self.clas0)

        # Relativo al algoritmo genetico
        self.n0 = n0
        self.probMut = probMut
        self.n_torneo = n_torneo
        self.n_elitismo = n_elitismo
        self.it = 0
        self.max_it = max_it
        # Probabilidades para los tamaños iniciales de cada cromosoma
        s = 0
        probs = []
        for i in range(10,0, -1):
            s += i
            probs.append(s)
        probs = np.array(probs[::-1])/sum(probs)
        self.poblacion = [Cromosoma(self, p = probs, evaluar = True) for _ in range(n0)]
        
    # Seleccion por torneo
    def seleccion(self):
        participantes = random.sample(self.poblacion, self.n_torneo)
        if self.maximizar: return max(participantes, key = lambda x : x.valor)
        else:  return min(participantes, key = lambda x : x.valor)
    
    # Cruce por punto
    def recombinacion(self, c1, c2):
        if c1.long == c2.long == 1:
            hijo1 = c1.genes + c2.genes
            hijo2 = c2.genes + c1.genes
        else:
            l = min(c1.long, c2.long)    
            corte = random.randint(1, l)
            hijo1 = c1.genes[:corte:] + c2.genes[corte:]
            hijo2 = c2.genes[:corte:] + c1.genes[corte:]
        return Cromosoma(self, hijo1), Cromosoma(self, hijo2)
    
    # Crear una nueva descendencia
    def generacion(self):
        elitismo = sorted(self.poblacion, key = lambda x : x.valor, reverse = self.maximizar)
        nueva_poblacion = elitismo[:self.n_elitismo]
        while len(nueva_poblacion) < self.n0:
            p1 = self.seleccion()
            p2 = self.seleccion()
            hijo1, hijo2 = self.recombinacion(p1, p2)
            if random.random() < self.probMut: hijo1.mutar()
            if random.random() < self.probMut: hijo2.mutar()
            hijo1.evaluar()
            hijo2.evaluar()
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        self.poblacion = nueva_poblacion
        self.it += 1

    def ejecutar(self, max_it = None):
        max_it = self.max_it if max_it is None else max_it
        for i in range(max_it): self.generacion()      

    def clasificar(self, G = None):
        if G is None: G = self.G0
        componentes = nx.connected_components(G)
        clas = -np.ones(len(self.datos))
        i = 0
        for CC in componentes:
            if len(CC) > 4: 
                clase = i
                i += 1
            else: clase = -1
            for nodo in list(CC): clas[nodo] = clase
        return clas
    
    def evaluar(self, G = None, clas = None, actualizar = False):
        if G is None and clas is None: clas = self.clas0
        elif clas is None: clas = self.clasificar(G)
        
        if self.metrica != 'adjusted_rand_score':    
            if self.metrica == 'calinski_harabasz_score':
                f_obj = calinski_harabasz_score
            elif self.metrica == 'silhouette_score':
                f_obj = silhouette_score
            elif self.metrica == 'davies_bouldin_score':
                f_obj = davies_bouldin_score
            else: raise Exception("No se ha introducido una métrica válida")  
            try:
                valor = f_obj(self.datos, clas)
                # Ignorando el ruido
                # valor = f_obj(self.datos[clas!=-1], clas[clas!=-1])
            except ValueError as e:
                    valor = 0
                    
        # En caso de utilizar el indice de rand, se compara con las etiquetas reales (supervisado)
        else:
            valor = adjusted_rand_score(self.target, clas)              
        mejora = (valor > self.valor) if self.maximizar else (valor < self.valor)
        # Si se produce una mejora, en caso de desearlo, se actualiza el mejor valor
        if mejora and actualizar:
            self.clas = clas
            self.G = G
            self.valor = valor
        return valor, mejora
    
    #Plot de la grafica asociada a un grafo (Solo si datos tiene 2 dimensiones)
    def plot(self, G = None, gama = 'gist_rainbow'):
        if G is None: G = self.G
        clas = self.clasificar(G)
        valor, _ = self.evaluar(G, clas, actualizar = False)
        # Mapa de colores
        cmap = cm.get_cmap(gama, int(max(clas)+1))
        color_map = []
        for i in clas:
            # Ruido en gris
            if i == -1: color_map.append((0.5, 0.5, 0.5, 1.0))
            else: color_map.append(cmap(int(i)))
        
        G_aux = G.copy()
        G_aux.add_nodes_from(range(len(self.datos)))
        _ = plt.figure(figsize=(10,7))
        nx.draw(G_aux, self.datos, nodelist = range(len(self.datos)), node_color= color_map, node_size=3, edge_color='skyblue')
        plt.title(f'{valor}')
        plt.show()
    

class Cromosoma():
    def __init__(self, ga, genes = None, p = None, evaluar = False):
        self.ga = ga
        if genes is None:
            self.genes = self.inicializar(p)
        else:
            self.genes = genes
        self.long = len(self.genes)
        self.valor = None
        self.k = self.ga.k0
        self.frontera = set()
        if evaluar: self.evaluar()
    
    def __repr__(self):
        return f"Genes: {self.genes}\n Valor: {self.valor}"
    
    # Inicializa aleatoriamente con una lista y los valores
    # Prioriza longitudes pequeñas pq son mas eficientes
    def inicializar(self, p):
        long = np.random.choice(range(1,11), p=p)
        genes = []
        for _ in range(long):
            eleccion = np.random.choice(movimientos)
            genes.append( (eleccion, eleccion.init()) )            
        return genes
    
    # Muta de acuerdo a 4 tipos posibles de mutuacion
    # Encoger es menos probable cuando el cromosoma es pequeño
    def mutar(self):
        tipo = ["Crece", "Encoge", "Criterio", "Umbral"]
        p = [1, 1*funcion_exponencial(self.long), 3, 2 ]  # Modificamos las probabilidades. Cuanto menor leng, menos probable es que encoja
        p = np.array(p)/sum(p)
        eleccion = np.random.choice(tipo, p=p)
        posiciones = range(self.long + int(eleccion == "Crece"))   # Si crece, puede estar tambien al finalm tiene una posicion mas
        indice = np.random.choice(posiciones)        
    
        if eleccion == "Umbral":
            mov, um = self.genes[indice]
            um = mov.mutar(um)
            self.genes[indice] = (mov, um)
        
        elif eleccion == "Criterio":
                mov = np.random.choice(movimientos)
                self.genes[indice] = (mov, mov.init())
        
        elif eleccion == "Crece":
            mov = np.random.choice(movimientos)
            self.genes = self.genes[:indice] + [(mov, mov.init())] + self.genes[indice:]
            self.long += 1
        
        else:
            self.genes.pop( indice )
            self.long -= 1

    def grafo(self):
        Gaux = self.ga.G0.copy()
        for mov, umbral in self.genes:
            mov( G = Gaux, umbral = umbral, cromosoma = self, vecinos = self.ga.vecinos, reverso = self.ga.reverso, M = self.ga.M)
        return Gaux

    def evaluar(self):
        Gaux = self.grafo()
        self.valor, mejora = self.ga.evaluar(Gaux, actualizar = True)
        if mejora: self.ga.best = self
            
# Para hacer menos probable que decrezca en valores pequeños
def funcion_exponencial(x, k=0.75):
    return np.clip(1 - np.exp(-k * (x-1)), 0, 1)


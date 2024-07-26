import networkx as nx
import numpy as np
import random

# Elimina aquellos elementos con grado 0 del grafo para agilizar
def quitar_ruido(G):
    ruido = [nodo for nodo, grado in G.degree() if grado == 0]
    G.remove_nodes_from(ruido)

# Declarar una lista de nodos como frontera, eliminando las aristas débiles
def frontera(G, nodos, cromosoma, reverso):
    for u in nodos:
        cromosoma.frontera.add(u)
        for v in list(G[u]):
            if (reverso[u,v] <= reverso[v,u]) and (cromosoma.k < reverso[v,u]): G.remove_edge(u,v)
            # Solo se va la arista si la conexion v, u es mas debil que k

# Definicion de los movimientos como clases callable
class cambiar_k():
    def __init__(self):
        self.name = "cambiar_k"
        self.desc = ("Añade o elimina aristas modificando el número de vecinos cercanos empleados k"
                     "Solo añade aristas que parten de elementos que no son frontera")
        
    def __repr__(self):
        return self.name
    
    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):
        k_viejo = cromosoma.k
        k_nuevo = k_viejo + umbral
        if -cromosoma.k < umbral < 0: # Eliminar las aristas u, v solo si el menor de los k es mas grande q el nuevo k 
            aristas_eliminar = [(u,v) for u in vecinos[:,0] for v in vecinos[u, k_nuevo+1: k_viejo +1] if (reverso[v,u] > k_nuevo or v in cromosoma.frontera) ]
            G.remove_edges_from(aristas_eliminar)
        
        elif 0 < umbral:    # Añadir las aristas excepto para las que parten de puntos frontera
            aristas_anadir = [(u,v, {'weight': M[u,v]}) for u in vecinos[:,0] for v in vecinos[u, k_viejo+1:k_nuevo +1] if (u not in cromosoma.frontera)  ]
            G.add_edges_from(aristas_anadir)       
        cromosoma.k = k_nuevo
        quitar_ruido(G)       

    def init(self, *args, **kargs):
        elegir = list(range(-5,6,1))
        elegir.pop(5) # Quitamos el 0 que no hace nada
        return np.random.choice(elegir)

    def mutar(self, valor, *args, **kargs):
        suma = np.random.choice([-2, -1, 1, 2])
        # Evita que el nuevo valor sea 0
        if valor == -suma and suma >0:
            return 1
        elif valor == -suma and suma >0:
            return -1            
        else: return valor + suma


class reducir_alpha():
    def __init__(self):
        self.name = "reducir_alpha"
        self.desc = ("Reduce el tamaño máximo de las aristas. Se eliminan las aristas con peso mayor que u por el tamaño de la mayor arista.")
        
    def __repr__(self):
        return self.name
    
    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):      
        alpha = max( [ M[i,vecinos[i, cromosoma.k]] for i in range(M.shape[0])])
        alpha2 = alpha*umbral               
        aristas_a_eliminar = np.where( alpha2 < M)
        G.remove_edges_from( zip(*aristas_a_eliminar) )
        quitar_ruido(G)       
    
    def init(self, *args, **kargs):
        return random.uniform(0.75, 1)

    def mutar(self, valor, *args, **kargs):
        return np.clip(valor + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1


class closeness():
    def __init__(self):
        self.name = "closeness"
        self.desc = ("Pone como frontera aquellos nodos que, tras normalizar, tienen un valor de "
                     "closeness_centrality menor que el umbral")
               
    def __repr__(self):
        return self.name

    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):
        CC = nx.connected_components(G)
        resultados = {}
        for cc in CC:
            cc, n = list(cc), len(cc)
            suma_dist = np.sum(M[np.ix_(cc,cc)], axis = 1)
            valores = ( n-1)**2 /( (len(G) -1) * suma_dist)
            for i in range(n):
                resultados[ cc[i] ] = valores[i]
                
        items, valores = np.array(list(resultados.keys())), np.array(list(resultados.values()))
                
        a, b = max(valores), min(valores)
        if a != b:
            valores = (valores-b) / (a-b)
            nodos_frontera = items[np.where( valores < umbral)]
            frontera(G, nodos_frontera, cromosoma, reverso)
            quitar_ruido(G)       

    def init(self, *args, **kargs):
        return random.uniform(0, 0.5)

    def mutar(self, valor, *args, **kargs):
        return np.clip(valor + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1


class distancia_media():
    def __init__(self):
        self.name = "distancia_media"
        self.desc = ("Elimina los nodos que, tras normalizar, tienen una distancia media de aristas mayor que el umbral")
               
    def __repr__(self):
        return self.name

    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):
        distancias_medias = np.array([ (u, np.sum([w for _,_,w in G.edges([u], data = "weight")]) ) for u in G.nodes])
        items = distancias_medias[:, 0].astype(int)
        valores = distancias_medias[:, 1] / np.array(G.degree)[:,1]
        a, b = max(valores), min(valores)
        if a != b:
            valores = (valores-b) / (a-b)
            nodos_frontera = items[np.where( valores > umbral)]
            frontera(G, nodos_frontera, cromosoma, reverso)
            quitar_ruido(G)       
    
    def init(self, *args, **kargs):
        return random.uniform(0.5, 1)

    def mutar(self, valor, *args, **kargs):
        return np.clip(valor + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1


class Jaccard():
    def __init__(self):
        self.name = "Jaccard"
        self.desc = ("Elimina las aristas cuyos nodos, tras normalizar, tienen un valor del "
                     "coeficiente de Jaccard menor que el umbral")
        
    def __repr__(self):
        return self.name

    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):
        c = {(u,v): j for u,v,j in nx.jaccard_coefficient(G, G.edges)}
        items, valores = np.array(list(c.keys())), np.array(list(c.values()))
        a, b = max(valores), min(valores)
        if a != b:
            valores = (valores-b) / (a-b)
            aristas_eliminar = items[valores < umbral]
            G.remove_edges_from(aristas_eliminar)
            quitar_ruido(G)       

    def init(self, *args, **kargs):
        return random.uniform(0, 0.5)

    def mutar(self, valor, *args, **kargs):
        return np.clip(valor + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1

class common_centrality():
    def __init__(self):
        self.name = "common neighbor centrality"
        self.desc = ("Elimina los nodos que, tras normalizar, tienen un valor de "
                     "common neighbor centrality menor que el umbral. "
                     "Tambien itera el valor de alpha")
               
    def __repr__(self):
        return self.name

    def __call__(self, G, umbral, cromosoma, vecinos, reverso, M, *args, **kargs):
        u, alpha = umbral
        c = {(u,v): alpha* len(list(nx.common_neighbors(G,u,v))) + (1-alpha)*len(G)/w for u,v,w in G.edges(data = 'weight')}
        items, valores = np.array(list(c.keys())), np.array(list(c.values()))
        a, b = max(valores), min(valores)
        if a != b:
            valores = (valores-b) / (a-b)
            aristas_eliminar = items[valores < u]
            G.remove_edges_from(aristas_eliminar)
            quitar_ruido(G)       
     
    def init(self, *args, **kargs):
        return (random.uniform(0, 0.5), random.uniform(0, 1))

    def mutar(self, valor, *args, **kargs):
        u, alpha = valor
        elec = np.random.choice([0,1,2])       
        if elec == 0:
            u2 = np.clip(u + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1
            return (u2, alpha)
        elif elec == 1:
            alpha2 = np.clip(alpha + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1
            return (u, alpha2)
        else:
            u2 = np.clip(u + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1
            alpha2 = np.clip(alpha + random.normalvariate(0, 0.2), 0, 1) # Que quede entre 0 y 1
            return (u2, alpha2)



movimientos = [cambiar_k(), reducir_alpha(), closeness(), distancia_media(), Jaccard(), common_centrality()]
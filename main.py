from Movimientos import *
from GPGAC import *

import numpy as np

nombres = ["2D002", "2D005", "2D01", "2D02", "5D002", "5D005", "5D01", "5D02", "10D002", "10D005", "10D01", "10D02"]
resultados = {}
for file in nombres:
    filename = f"Datasets/{file}.dat"
    lista = []
    with open(filename, "r") as f:
        linea = f.readline()
        linea = f.readline()
        while linea != "":
            puntos = linea.split(" ")
            puntos = [float(x) for x in puntos]
            lista.append(puntos)
            linea = f.readline()

    datos = np.array(lista)

    ga = GPGAC(datos, k = 8)
    ga.ejecutar()

    if file[0] == '2':
        ga.plot()
        plt.show()

    r = {'GPGAC': ga, 'clas': ga.clas,
     'ch': calinski_harabasz_score(ga.datos, ga.clas),
     'sil': silhouette_score(ga.datos, ga.clas),
     'db': davies_bouldin_score(ga.datos, ga.clas),
    }

    if len(set(ga.clas[ga.clas >= 0])) > 1:
        r['ch2'] = calinski_harabasz_score(ga.datos[ga.clas >= 0], ga.clas[ga.clas >= 0])
        r['sil2'] = silhouette_score(ga.datos[ga.clas >= 0], ga.clas[ga.clas >= 0])
        r['db2'] = davies_bouldin_score(ga.datos[ga.clas >= 0], ga.clas[ga.clas >= 0])
    else:
        r['ch2'] = 0
        r['sil2'] = 0
        r['db2'] = float('inf')

    resultados[file] = r

# En resultados se guarda un diccionario con los datos de la ejecucion sobre cada dataset

# Para representar las graficas juntas
def plot_en_ax(datos, clas, ax, titulo):
    ax.scatter( datos[ clas >= 0][:,0], datos[clas >= 0][:,1], s = 4, c = clas[clas >= 0], cmap = 'gist_rainbow')
    ax.scatter( datos[ clas < 0 ][:,0], datos[clas <  0][:,1], s = 4, color = 'grey')
    ax.set_title(titulo)

fig, axs = plt.subplots(2, 2, figsize = (15,10))


r = resultados["2D002"]
plot_en_ax( r['GPGAC'].datos, r['clas'], axs[0,0], f"Valor del ínidice de Calinski-Harabasz para 2D002: {r['ch']: .3f}")

r = resultados["2D005"]
plot_en_ax( r['GPGAC'].datos, r['clas'], axs[0,1], f"Valor del ínidice de Calinski-Harabasz para 2D005: {r['ch']: .3f}")

r = resultados["2D01"]
plot_en_ax( r['GPGAC'].datos, r['clas'], axs[1,0], f"Valor del ínidice de Calinski-Harabasz para 2D01: {r['ch']: .3f}")

r = resultados["2D02"]
plot_en_ax( r['GPGAC'].datos, r['clas'], axs[1,1], f"Valor del ínidice de Calinski-Harabasz para 2D02: {r['ch']: .3f}")


fig.tight_layout()
plt.show()
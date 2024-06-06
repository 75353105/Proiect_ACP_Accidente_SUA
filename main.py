import numpy as np

from AnalizaComponentelorPrincipale import *
from grafice import *


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


mortalitate_SUA = pd.read_csv("date.csv", index_col='US State')
#print(mortalitate_SUA.all)


variabile_observate = list(mortalitate_SUA)[1:]
print(variabile_observate)
model = ACP(mortalitate_SUA,variabile_observate)


etichete = ["C" + str(i) for i in range(1, len(model.vect_valProprii) + 1)]
print(model.tabel_varianta(etichete))

#Varianta
model.tabel_varianta(etichete).to_csv("Distributia_variantei.csv")
grafic_varianta(model)

#Corelatiile
tabel_corelatii = model.tabel(model.corelatiiIntreXsiC(), variabile_observate,
            etichete, "TabelCorelatii.csv")
graficCorelatii(tabel_corelatii)
#graficCorelatii(tabel_corelatii, "C1", "C3")
corelograma(tabel_corelatii)

#Componentele
tabel_componente = model.tabel(model.componente_principale ,mortalitate_SUA.index, etichete, "TabelComponente.csv");
graficInstante(tabel_componente,"C1","C2", "Grafic de componente")


#Scoruri
tabel_scoruri = model.tabel(model.componente_principale/np.sqrt(model.vect_valProprii),
                            mortalitate_SUA.index, etichete, "Scoruri.csv")
graficInstante(tabel_scoruri)

#Cosinusuri - ungiul sa se apropie mai mult de 0, daca e in stanga sus sa se apropie de 180
cPatrat = model.componente_principale * model.componente_principale
cosin = np.transpose(cPatrat.T / np.sum(cPatrat,axis=1))
tabel_cosin = model.tabel(cosin, mortalitate_SUA.index, etichete, "Cosin.csv")

#Contributii
contributii = cPatrat * 100/np.sum(cPatrat, axis=0)
tabel_contributii = model.tabel(contributii, mortalitate_SUA.index, etichete, "Contributii.csv")

#Comunalitati
r2 = model.corelatiiIntreXsiC() * model.corelatiiIntreXsiC()
comunalitati = np.cumsum(r2, axis=1)
tabel_comunalitati = model.tabel(tabel_corelatii, tabel_corelatii.index, tabel_contributii.columns, "Comunalitati.csv")

corelograma(tabel_comunalitati, valMax=0, titlu="Grafic comunalitati")
#arataGrafice()




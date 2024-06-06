import numpy as np
import pandas as pd

import grafice
import functii
from AnalizaComponentelorPrincipale import *
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from main import *

#testul Bartlett de sfericitate si validitatea modelului
date_test_Bartlett= model.variabile_observate
valoare_chi_patrat,p_value=calculate_bartlett_sphericity(date_test_Bartlett)

print(f"valoare_chi_patrat: {valoare_chi_patrat}")
print(f"valoare_p: {p_value}")

if(p_value<0.05):
    print("Ipoteza nula este respina. Se poate merge mai departe cu analiza.")
else:
    print("Ipoteza nula nu poatae fi respinsa. Va trebui cautat un alt set de date.")

kmo=calculate_kmo(date_test_Bartlett)
t_kmo=pd.DataFrame(
    {
     "KMO":np.append(kmo[0],kmo[1])
    }, index=variabile_observate+["KMO total"]
)
t_kmo.to_csv("kmo.csv")
corelograma(t_kmo,valMin=0,valMax=1,titlu="Index KMO")

#Observam ca setul de date este potrivit si se poate merge mai departe
#Cream modelul pentru analiza factoriala

num_factors=9
model_af=FactorAnalyzer(n_factors=num_factors,rotation='varimax')
model_af.fit(model.variabile_observate)

#Analiza variantei
varianta_af=model_af.get_factor_variance()
etichete_factori_af=["F" + str(i) for i in range(1,10)]
tabel_varianta_af=pd.DataFrame(
    {
        "Varianta":varianta_af[0],
        "Procentul variantei":varianta_af[1]*100,
        "Procentul cumulat":varianta_af[2]*100
    },etichete_factori_af
)
tabel_varianta_af.to_csv("varianta_af.csv")
alpha=varianta_af[0]
criterii=functii.calcul_criterii(alpha)
grafic_varianta_af(alpha,criterii,eticheta_x="Factor")
#arataGrafice()

#Numar de factori semnificativi
k=int(np.nanmean(criterii))

#Calcularea corelatiilor intre variabilele observate si factori

#cu rotatie
corelatii_factoriale=model_af.loadings_
tabel_corelatii_af1=pd.DataFrame(corelatii_factoriale,columns=variabile_observate,index=etichete_factori_af)
tabel_corelatii_af1.to_csv("Corelatii_factoriale_cu_rotatie.csv")
corelograma(tabel_corelatii_af1)


#fara rotatie
model_af_fara_rotatie=FactorAnalyzer(n_factors=num_factors,rotation=None)
model_af_fara_rotatie.fit(model.variabile_observate)
corelatii_factoriale_fara_rotatie=model_af_fara_rotatie.loadings_
tabel_corelatii_af2=pd.DataFrame(corelatii_factoriale_fara_rotatie,columns=variabile_observate,index=etichete_factori_af)
tabel_corelatii_af2.to_csv("Corelatii_factoriale_fara_rotatie.csv")
corelograma(tabel_corelatii_af2)


#Calcularea scorurilor factoriale
#Cu rotatie
scoruri_rotatie=model_af.transform(model.variabile_observate)
tabel_scoruri_rotatie=pd.DataFrame(scoruri_rotatie.T,columns=mortalitate_SUA.index,index=etichete_factori_af)
tabel_scoruri_rotatie.to_csv("Tabel_scoruri_rotatie.csv")
for j in range(2,k+1):
    graficInstante(tabel_scoruri_rotatie.T,variabila1="F1",variabila2="F"+str(j),corelatii=True)

#Fara rotatie
scoruri_fara_rotatie=model_af_fara_rotatie.transform(model.variabile_observate)
tabel_scoruri_fara_rotatie=pd.DataFrame(scoruri_fara_rotatie.T,columns=mortalitate_SUA.index,index=etichete_factori_af)
tabel_scoruri_fara_rotatie.to_csv("Tabel_scoriru_fara_rotatie.csv")
for j in range(2,k+1):
    graficInstante(tabel_scoruri_fara_rotatie.T,variabila1="F1",variabila2="F"+str(j),corelatii=True)

#Analiza comunalitatii si a variantei specifice
comunalitate=model_af.get_communalities()
unicitate=model_af.get_uniquenesses()
tabel_comunalitati_af=pd.DataFrame(
    {
        "Comunalitate":comunalitate,
        "Varianta specifica":unicitate
    },variabile_observate
)
tabel_comunalitati_af.to_csv("Tabel_comunalitati_af.csv")

corelograma(tabel_comunalitati_af,valMax=0,titlu="Grafic comunalitati analiza factoriala")

arataGrafice()



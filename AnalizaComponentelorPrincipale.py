import numpy as np
import pandas as pd

class ACP:
    def __init__(self, mortalitate_SUA, variabile_observate, std=True):
        assert isinstance(mortalitate_SUA, pd.DataFrame)

        #Completarea datelor goale
        for i in mortalitate_SUA.columns:
            if mortalitate_SUA[i].isna().any():
                if pd.api.types.is_numeric_dtype(mortalitate_SUA[i]):
                    mortalitate_SUA[i].fillna(mortalitate_SUA[i].mean(), inplace=True)
                else:
                    mortalitate_SUA[i].fillna(mortalitate_SUA[i].mode()[0], inplace=True)

        self.__variabile_observate = mortalitate_SUA[variabile_observate].values

        # Centram datele
        self.__medie_aritmetica = np.mean(self.__variabile_observate, axis=0)
        self.__date_centrate = self.__variabile_observate - self.__medie_aritmetica
        if std:
            self.__date_centrate =  self.__date_centrate/np.std(self.__variabile_observate, axis=0)

        # Calculam matricea de covariatie
        self.__matrice_covariatie = np.cov(self.__date_centrate, rowvar=False)

        # Calculam vectorul cu valorile proprii si matrice cu vectorii proprii asezati pe coloane
        vect_valProprii, matr_vectProprii = np.linalg.eigh(self.__matrice_covariatie)

        # acestea se sorteaza in ordine descrescatoare
        indici_sortati = np.flipud(np.argsort(vect_valProprii))
        self.__vect_valProprii = vect_valProprii[indici_sortati]
        self.__matr_vectProprii = matr_vectProprii[:, indici_sortati]

        # regularizarea vectorilor proprii
        for j in range(len(self.__vect_valProprii)):
            minim = np.min(self.__matr_vectProprii[:, j])
            maxim = np.max(self.__matr_vectProprii[:, j])
            if np.abs(minim) > np.abs(maxim):
                self.__matr_vectProprii[:, j] *= -1  # inmultirea unui vector propriu cu un scalar nu modifica calitatea de vector propriu

        self.__n, self.__m = np.shape(self.__date_centrate)
        self.__componente_principale = self.__date_centrate@self.__matr_vectProprii

        #Initializam si procentele cumulate ale variantei
        self.__procente_cumulate_varianta = self.__vect_valProprii * 100 / np.sum(self.__vect_valProprii)


    def tabel_varianta(self, etichete):
        return pd.DataFrame(data={
            "Varianta: ":self.__vect_valProprii,
            "Procent varianta: ": self.__procente_cumulate_varianta,
            "Varianta cumulata: ": np.cumsum(self.__vect_valProprii),
            "Procent cumulat: ": np.cumsum(self.__procente_cumulate_varianta)
        }, index=etichete)

    def tabel(self, x, linii, coloane, fisierul):
        if fisierul is not None:
            tabel = pd.DataFrame(x, linii, coloane)
            tabel.to_csv(fisierul)
            return tabel
        else:
            return 0

    def criteriulProcentuluiVariantaExplicata(self, procentMin = 80):
        return np.where(self.__procente_cumulate_varianta > procentMin)[0][0] + 1

    def corelatiiIntreXsiC(self, std=True):
        if std:
            return self.matr_vectProprii * np.sqrt(self.vect_valProprii)
        else:
            return np.corrcoef(self.__date_centrate, rowvar=False)[:self.__m, self.__m:]

    def criteriulKaiser(self, std):
        if std:
            kaiser = len(np.where(self.__vect_valProprii > 1)[0])
        else:
            kaiser = None
        return kaiser

    def criteriulCattell(self):
        epsilon = self.__vect_valProprii[:(self.__m-1)] - self.__vect_valProprii[1:]
        sigma = epsilon[:(self.__m-2)] - epsilon[1:]
        valori_negative = sigma < 0
        if any(valori_negative):
            pozitie = np.where(valori_negative)[0][0]
            return pozitie+1
        else:
            return None

    @property
    def procente_cumulate_varianta(self):
        return self.__procente_cumulate_varianta

    @property
    def n(self):
        return self.__n


    @property
    def m(self):
        return self.__m

    @property
    def variabile_observate(self):
        return self.__variabile_observate

    @property
    def medie_aritmetica(self):
        return self.__medie_aritmetica

    @property
    def date_centrate(self):
        return self.__date_centrate

    @property
    def matrice_covariatie(self):
        return self.__matrice_covariatie

    @property
    def vect_valProprii(self):
        return self.__vect_valProprii

    @property
    def matr_vectProprii(self):
        return self.__matr_vectProprii

    @property
    def componente_principale(self):
        return self.__componente_principale



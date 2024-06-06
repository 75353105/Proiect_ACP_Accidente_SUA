import matplotlib.pyplot as grafic
from AnalizaComponentelorPrincipale import ACP
import numpy as np
import seaborn as sb

def grafic_varianta(model, std = True):
    figura = grafic.figure(figsize=(14,8))
    axa = figura.add_subplot(1,1,1)
    axa.set_title("Plot varianta + componente", fontdict={"fontsize": 20, "color":"c"})
    axa.set_xlabel("Componenta")
    axa.set_ylabel("Varianta")
    m = len(model.vect_valProprii)
    x = np.arange(1, m+1)
    axa.set_xticks(x)
    axa.plot(x, model.vect_valProprii, c="c")
    axa.scatter(x, model.vect_valProprii, c="b")

    if model.criteriulKaiser(std) is not None:
        axa.axhline(1, c='y', label='Criteriul Kaiser')

    if model.criteriulCattell() is not None:
        axa.axhline(model.vect_valProprii[model.criteriulCattell()-1], c='m', label='Criteriul Cattell')
    axa.axhline(model.vect_valProprii[model.criteriulProcentuluiVariantaExplicata()-1], c='r', label = 'Criteriul procentului de varianta explicata')
    grafic.legend()
    assert isinstance(model, ACP)
    assert isinstance(figura, grafic.Figure)
    assert isinstance(axa,grafic.Axes)

def grafic_varianta_af(alpha, criterii, procent_minimal=70,eticheta_x="Componenta"):
    fig = grafic.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, grafic.Axes)
    ax.set_title("Plot varianta analiza factorilor + Componente", fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(eticheta_x)
    ax.set_ylabel("Varianta")
    x = np.arange(1, len(alpha) + 1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)
    ax.axhline(alpha[criterii[0] - 1], c="m", label="Varianta minimala:" + str(procent_minimal) + "%")
    if not np.isnan(criterii[1]):
        ax.axhline(1, c="c", label="Kaiser")
    if not np.isnan(criterii[2]):
        ax.axhline(alpha[criterii[2] - 1], c="g", label="Cattell")
    ax.legend()
    grafic.savefig("Plot_varianta")

def graficCorelatii(t,variabila1="C1",variabila2="C2"):
    figura = grafic.figure(figsize=(11,9))
    assert isinstance(figura, grafic.Figure);
    axa = figura.add_subplot(1,1,1,aspect=1)
    assert isinstance(axa,grafic.Axes)
    axa.set_title("Cercul corelatiilor factoriale", fontdict={"fontsize":20, "color":"c"})
    axa.set_xlabel(variabila1, fontdict={"fontsize":20, "color":"c"})
    axa.set_ylabel(variabila2, fontdict={"fontsize":20, "color":"c"})
    axa.set_aspect(1)
    u = np.arange(0, np.pi*2, 0.01)
    axa.plot(np.cos(u), np.sin(u), c="c")
    axa.axhline(0)
    axa.axvline(0)

    axa.scatter(t[variabila1], t[variabila2], c="r")
    for i in range(len(t)):
        axa.text(t[variabila1].iloc[i], t[variabila2].iloc[i], t.index[i])

def corelograma(t,valMin=-1, valMax=1, titlu="Corelograma corelații dintre variabilele observate și componente"):
    figura = grafic.figure(figsize=(11,9))
    assert isinstance(figura,grafic.Figure)
    axa = figura.add_subplot(1, 1, 1, aspect=1)
    assert isinstance(axa, grafic.Axes)
    axa.set_title(titlu, fontdict={"fontsize":20, "color":"c"})
    axa2 = sb.heatmap(t, vmin=valMin, vmax=valMax, cmap="RdYlBu", annot=True, ax=axa)
    axa2.set_xticklabels(t.columns, rotation=30, ha="right")

def arataGrafice():
    grafic.show()

def graficInstante(t,variabila1="C1",variabila2="C2", titlu = "Grafic de scoruri",corelatii=False):
    figura = grafic.figure(figsize=(15,14))
    assert isinstance(figura, grafic.Figure);
    axa = figura.add_subplot(1,1,1,aspect='auto')
    assert isinstance(axa,grafic.Axes)
    axa.set_title(titlu, fontdict={"fontsize":20, "color":"c"})
    axa.set_xlabel(variabila1, fontdict={"fontsize":40, "color":"c"})
    axa.set_ylabel(variabila2, fontdict={"fontsize":40, "color":"c"})
    axa.set_aspect(1)
    if corelatii:
        tetha=np.arange(0,np.pi*2,0.01)
        axa.plot(np.cos(tetha),np.sin(tetha),c="m")
        axa.plot(0.65*np.cos(tetha),0.65*np.sin(tetha),c="g")
    axa.scatter(t[variabila1], t[variabila2], c="r",alpha=0.5)
    for i in range(len(t)):
        axa.text(t[variabila1].iloc[i], t[variabila2].iloc[i], t.index[i])


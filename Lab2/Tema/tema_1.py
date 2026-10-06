# tema 1 - laborator 2 - simularea variabilelor aleatoare

import csv
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # salvare grafice in fisiere, fara fereastra
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------
# exercitiul 1 - esantion aleator fara repetitie dintr-un fisier csv
# ---------------------------------------------------------------
def citeste_lista_csv(cale):
    # citeste prima coloana din fisierul csv (se ignora antetul)
    with open(cale, newline="", encoding="utf-8") as f:
        cititor = csv.reader(f)
        next(cititor)  # sare peste antet
        return [rand[0] for rand in cititor if rand]


def exercitiul_1(cale="studenti.csv", k=3):
    print("=== exercitiul 1 ===")
    lista = citeste_lista_csv(cale)
    # random.sample alege k elemente distincte
    esantion = random.sample(lista, k)
    print(f"{k} studenti alesi aleator, fara repetitie:")
    for nume in esantion:
        print("  -", nume)
    print()


# ---------------------------------------------------------------
# exercitiul 2 - jocul cu moneda si zarul
# ---------------------------------------------------------------
# a) N = numarul de aruncari pana la prima stema => N ~ Geom(p),
#    cu P(N = n) = (1 - p)^(n - 1) * p, n = 1, 2, ...
#    (p = probabilitatea de aparitie a stemei, p = 0.5 la moneda corecta)

def joc(p=0.5):
    # b) simuleaza un joc; returneaza (N, S)
    # S = suma pe care al doilea jucator o da primului
    n = 0
    s = 0.0
    while True:
        n += 1
        if rng.random() < p:          # a picat stema
            z = rng.integers(1, 7)    # aruncare cu zarul
            s += z - 3
            break
        else:                         # a picat banul
            s -= 0.5                  # primul ii da 0.5 $ celui de-al doilea
    return n, s


def simuleaza(p, nr_jocuri=100000):
    rezultate = [joc(p) for _ in range(nr_jocuri)]
    N = np.array([r[0] for r in rezultate])
    S = np.array([r[1] for r in rezultate])
    return N, S


def exercitiul_2():
    print("=== exercitiul 2 ===")
    print("a) N urmeaza o distributie geometrica Geom(p)")

    n, s = joc(0.5)
    print(f"b) exemplu de joc: N = {n}, S = {s}")

    cazuri = [(0.5, "moneda corecta, p = 0.5"),
              (0.3, "moneda masluita, p = 0.3"),
              (0.7, "moneda masluita, p = 0.7")]

    fig, axe = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (p, titlu) in zip(axe, cazuri):
        N, S = simuleaza(p)
        # valoarea teoretica: E[S] = E[z - 3] - 0.5 * E[N - 1]
        teoretic = 0.5 - 0.5 * (1 - p) / p
        print(f"c/d) {titlu}: media lui S ~ {S.mean():.4f} "
              f"(teoretic {teoretic:.4f}), media lui N ~ {N.mean():.4f} "
              f"(teoretic {1 / p:.4f})")
        ax.hist(S, bins=60, density=True, color="steelblue", edgecolor="white")
        ax.axvline(S.mean(), color="red", linestyle="--",
                   label=f"media = {S.mean():.2f}")
        ax.set_title(titlu)
        ax.set_xlabel("S")
        ax.set_ylabel("densitate")
        ax.legend()
    plt.tight_layout()
    plt.savefig("ex2_histograme.png", dpi=120)
    plt.close()
    print("histogramele au fost salvate in ex2_histograme.png\n")


# ---------------------------------------------------------------
# exercitiul 3 - frizeria
# ---------------------------------------------------------------
# probabilitatile 3/13, 6/13, 4/13 sunt proportionale cu vitezele de
# servire: un frizer mai rapid termina mai repede si preia mai multi clienti,
# deci fractiunea de clienti servita de fiecare este lambda_i / suma lambda.
# X este o mixtura de distributii exponentiale.

def exercitiul_3(nr_valori=10000):
    print("=== exercitiul 3 ===")
    lambde = np.array([3.0, 6.0, 4.0])
    probabilitati = lambde / lambde.sum()  # 3/13, 6/13, 4/13

    # se alege frizerul pentru fiecare client, apoi timpul de servire
    frizer = rng.choice(3, size=nr_valori, p=probabilitati)
    X = rng.exponential(scale=1 / lambde[frizer])

    # valori teoretice pentru comparatie
    medie_t = np.sum(probabilitati / lambde)
    m2_t = np.sum(probabilitati * 2 / lambde ** 2)
    std_t = np.sqrt(m2_t - medie_t ** 2)

    print(f"media estimata   = {X.mean():.4f} (teoretic {medie_t:.4f})")
    print(f"deviatia estimata = {X.std(ddof=1):.4f} (teoretic {std_t:.4f})")

    # grafic aproximativ al densitatii
    plt.figure(figsize=(7, 4))
    try:
        import arviz as az
        az.plot_kde(X)
    except ImportError:
        # varianta de rezerva daca arviz nu este instalat
        grila = np.linspace(0, X.max(), 400)
        plt.plot(grila, stats.gaussian_kde(X)(grila))
    plt.title("densitatea aproximativa a lui X")
    plt.xlabel("timp de servire (ore)")
    plt.ylabel("densitate")
    plt.tight_layout()
    plt.savefig("ex3_densitate.png", dpi=120)
    plt.close()
    print("graficul a fost salvat in ex3_densitate.png")


# ---------------------------------------------------------------
# rulare
# ---------------------------------------------------------------
if __name__ == "__main__":
    exercitiul_1()
    exercitiul_2()
    exercitiul_3()

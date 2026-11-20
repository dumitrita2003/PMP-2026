import numpy as np

# stari ascunse: camerele 1, 2, 3, 4 (indexate 0, 1, 2, 3)
nume_camere = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

# observatii: 0 = red, 1 = green, 2 = blue
mapare_culori = {"red": 0, "green": 1, "blue": 2}
secventa_culori = ["red", "green", "blue", "blue", "red"]
secventa_indici = [mapare_culori[culoare] for culoare in secventa_culori]

# distributia initiala uniforma pe cele 4 camere
distributie_initiala = np.array([0.25, 0.25, 0.25, 0.25])

# matricea de emisie B (4 camere x 3 culori)
matrice_emisie = np.array([
    [0.7, 0.2, 0.1],  # camera 1
    [0.2, 0.6, 0.2],  # camera 2
    [0.1, 0.3, 0.6],  # camera 3
    [0.5, 0.2, 0.3],  # camera 4
])


def ruleaza_viterbi(prob_initiala, matrice_tranzitie, matrice_observatii, observatii):
    numar_pasi = len(observatii)
    numar_stari = len(prob_initiala)

    delta = np.zeros((numar_pasi, numar_stari))
    psi = np.zeros((numar_pasi, numar_stari), dtype=int)

    # initializare la pasul t = 0
    delta[0] = prob_initiala * matrice_observatii[:, observatii[0]]

    # recursie pentru t = 1 ... T-1
    for pas_timp in range(1, numar_pasi):
        for stare_curenta in range(numar_stari):
            probabilitati_venire = delta[pas_timp - 1] * matrice_tranzitie[:, stare_curenta]
            psi[pas_timp, stare_curenta] = int(np.argmax(probabilitati_venire))
            delta[pas_timp, stare_curenta] = (
                matrice_observatii[stare_curenta, observatii[pas_timp]]
                * np.max(probabilitati_venire)
            )

    # backtracking pentru reconstituirea traseului optim
    traseu_indici = np.zeros(numar_pasi, dtype=int)
    traseu_indici[-1] = int(np.argmax(delta[-1]))
    probabilitate_comuna = float(np.max(delta[-1]))

    for pas_timp in range(numar_pasi - 2, -1, -1):
        traseu_indici[pas_timp] = psi[pas_timp + 1, traseu_indici[pas_timp + 1]]

    traseu_camere = [nume_camere[idx] for idx in traseu_indici]
    return traseu_camere, probabilitate_comuna


# --- punctul (a) ---
# ramane pe loc cu 0.2, stanga cu 0.4, dreapta cu 0.4 (dispunere circulara)
matrice_tranzitie_a = np.array([
    [0.2, 0.4, 0.0, 0.4],  # din camera 1 -> 1, 2, 3, 4
    [0.4, 0.2, 0.4, 0.0],  # din camera 2
    [0.0, 0.4, 0.2, 0.4],  # din camera 3
    [0.4, 0.0, 0.4, 0.2],  # din camera 4
])

secventa_a, prob_a = ruleaza_viterbi(
    distributie_initiala, matrice_tranzitie_a, matrice_emisie, secventa_indici
)
print("=== SUBIECTUL 1 (a) ===")
print("Cea mai probabila secventa de camere:", " -> ".join(secventa_a))
print(f"Probabilitatea comuna P(S*, O): {prob_a:.8f}")

# --- punctul (b) ---
# nu mai ramane pe loc (0.0), stanga cu 0.5, dreapta cu 0.5
matrice_tranzitie_b = np.array([
    [0.0, 0.5, 0.0, 0.5],  # din camera 1
    [0.5, 0.0, 0.5, 0.0],  # din camera 2
    [0.0, 0.5, 0.0, 0.5],  # din camera 3
    [0.5, 0.0, 0.5, 0.0],  # din camera 4
])

secventa_b, prob_b = ruleaza_viterbi(
    distributie_initiala, matrice_tranzitie_b, matrice_emisie, secventa_indici
)
print("\n=== SUBIECTUL 1 (b) ===")
print("Cea mai probabila secventa de camere:", " -> ".join(secventa_b))
print(f"Probabilitatea comuna P(S*, O): {prob_b:.8f}")
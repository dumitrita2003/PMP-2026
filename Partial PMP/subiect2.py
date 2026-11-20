import numpy as np
import matplotlib.pyplot as plt

# 1. desenarea retelelor markov asociate celor doua factorizari
figura, axe = plt.subplots(1, 2, figsize=(11, 4.5))

# graf (a): lantul A - B - C - D - E
pozitii_a = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (2.0, 0.0),
    "D": (3.0, 0.0),
    "E": (4.0, 0.0),
}
muchii_a = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]

for nod_start, nod_final in muchii_a:
    axe[0].plot(
        [pozitii_a[nod_start][0], pozitii_a[nod_final][0]],
        [pozitii_a[nod_start][1], pozitii_a[nod_final][1]],
        color="black",
        linewidth=2,
        zorder=1,
    )

for nume_nod, (coord_x, coord_y) in pozitii_a.items():
    axe[0].scatter(coord_x, coord_y, s=900, color="#d9eaf7", edgecolors="black", linewidth=1.5, zorder=2)
    axe[0].text(coord_x, coord_y, nume_nod, ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)

axe[0].set_title("Subiect 2 (a): A - B - C - D - E")
axe[0].set_xlim(-0.6, 4.6)
axe[0].set_ylim(-1.0, 1.0)
axe[0].axis("off")

# graf (b): triunghiul {A, B, C} + muchiile B-D si C-E
pozitii_b = {
    "A": (1.5, 1.5),
    "B": (0.5, 0.3),
    "C": (2.5, 0.3),
    "D": (0.5, -1.0),
    "E": (2.5, -1.0),
}
muchii_b = [("A", "B"), ("B", "C"), ("A", "C"), ("B", "D"), ("C", "E")]

for nod_start, nod_final in muchii_b:
    axe[1].plot(
        [pozitii_b[nod_start][0], pozitii_b[nod_final][0]],
        [pozitii_b[nod_start][1], pozitii_b[nod_final][1]],
        color="black",
        linewidth=2,
        zorder=1,
    )

for nume_nod, (coord_x, coord_y) in pozitii_b.items():
    axe[1].scatter(coord_x, coord_y, s=900, color="#d9eaf7", edgecolors="black", linewidth=1.5, zorder=2)
    axe[1].text(coord_x, coord_y, nume_nod, ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)

axe[1].set_title("Subiect 2 (b): Clica {A, B, C}, muchii B-D si C-E")
axe[1].set_xlim(-0.2, 3.2)
axe[1].set_ylim(-1.6, 2.1)
axe[1].axis("off")

plt.tight_layout()
plt.savefig("subiect2_retele_markov.png", dpi=150)
plt.close()

# 2. verificare numerica a simplificarilor pe potentiale pozitive aleatoare
rng = np.random.default_rng(42)

# verificare (a): P(A | B, C) = phi_1(A, B) / sum_{A'} phi_1(A', B)
phi_1 = rng.uniform(0.1, 2.0, size=(2, 2))
phi_2 = rng.uniform(0.1, 2.0, size=(2, 2))
phi_3 = rng.uniform(0.1, 2.0, size=(2, 2))
phi_4 = rng.uniform(0.1, 2.0, size=(2, 2))

distributie_comuna_a = np.einsum("ab,bc,cd,de->abcde", phi_1, phi_2, phi_3, phi_4)
distributie_comuna_a /= np.sum(distributie_comuna_a)

marginala_abc = np.sum(distributie_comuna_a, axis=(3, 4))
marginala_bc = np.sum(marginala_abc, axis=0)
cond_a_dat_bc_exact = marginala_abc / marginala_bc[None, :, :]
cond_a_dat_bc_formula = (phi_1 / np.sum(phi_1, axis=0, keepdims=True))[:, :, None]

print("=== SUBIECTUL 2 (a) ===")
print("Formula simplificata: P(A | B, C) = phi_1(A, B) / sum_{A'} phi_1(A', B)")
print("Eroare maxima verificare numerica:", np.max(np.abs(cond_a_dat_bc_exact - cond_a_dat_bc_formula)))

# verificare (b): P(D | A, B, C, E) = psi_2(B, D) / sum_{D'} psi_2(B, D')
psi_1 = rng.uniform(0.1, 2.0, size=(2, 2, 2))
psi_2 = rng.uniform(0.1, 2.0, size=(2, 2))
psi_3 = rng.uniform(0.1, 2.0, size=(2, 2))

distributie_comuna_b = np.einsum("abc,bd,ce->abcde", psi_1, psi_2, psi_3)
distributie_comuna_b /= np.sum(distributie_comuna_b)

marginala_abce = np.sum(distributie_comuna_b, axis=3)
cond_d_dat_abce_exact = distributie_comuna_b / marginala_abce[:, :, :, None, :]
cond_d_dat_abce_formula = (psi_2 / np.sum(psi_2, axis=1, keepdims=True))[None, :, None, :, None]

print("\n=== SUBIECTUL 2 (b) ===")
print("Formula simplificata: P(D | A, B, C, E) = psi_2(B, D) / sum_{D'} psi_2(B, D')")
print("Eroare maxima verificare numerica:", np.max(np.abs(cond_d_dat_abce_exact - cond_d_dat_abce_formula)))
print("\nGraficul 'subiect2_retele_markov.png' a fost salvat.")
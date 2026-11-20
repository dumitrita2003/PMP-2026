import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# prior initial: theta ~ Beta(2, 8)
alpha_prior = 2
beta_prior = 8

# --- punctul (a): primul lot -> 5 defecte din 20 produse ---
defecte_lot1 = 5
total_lot1 = 20

alpha_post_a = alpha_prior + defecte_lot1
beta_post_a = beta_prior + (total_lot1 - defecte_lot1)
media_post_a = alpha_post_a / (alpha_post_a + beta_post_a)

print("=== SUBIECTUL 3 (a) ===")
print(f"Distributia a posteriori dupa lotul 1: Beta({alpha_post_a}, {beta_post_a})")
print(f"Media a posteriori: {media_post_a:.6f} (exact: {alpha_post_a}/{alpha_post_a + beta_post_a})")

# grafic densitati a priori si a posteriori pentru (a)
grila_theta = np.linspace(0, 1, 500)
densitate_prior = beta.pdf(grila_theta, alpha_prior, beta_prior)
densitate_post_a = beta.pdf(grila_theta, alpha_post_a, beta_post_a)

plt.figure(figsize=(8, 4.5))
plt.plot(grila_theta, densitate_prior, label="Prior: Beta(2, 8)", linestyle="--", linewidth=2)
plt.plot(grila_theta, densitate_post_a, label=f"Posterior (a): Beta({alpha_post_a}, {beta_post_a})", linewidth=2)
plt.axvline(media_post_a, color="tab:orange", linestyle=":", label=f"Media posterior = {media_post_a:.3f}")
plt.title("Subiectul 3 (a): Densitatea a priori vs. a posteriori")
plt.xlabel("theta (probabilitatea de defect)")
plt.ylabel("Densitate de probabilitate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("subiect3_densitati.png", dpi=150)
plt.close()
print("Graficul 'subiect3_densitati.png' a fost salvat.")

# --- punctul (b): actualizare secventiala cu al doilea lot -> 3 defecte din 15 produse ---
defecte_lot2 = 3
total_lot2 = 15

alpha_post_b = alpha_post_a + defecte_lot2
beta_post_b = beta_post_a + (total_lot2 - defecte_lot2)
media_post_b = alpha_post_b / (alpha_post_b + beta_post_b)

print("\n=== SUBIECTUL 3 (b) ===")
print(f"Distributia a posteriori dupa ambele loturi: Beta({alpha_post_b}, {beta_post_b})")
print(f"Media a posteriori: {media_post_b:.6f} (exact: {alpha_post_b}/{alpha_post_b + beta_post_b})")

# --- punctul (c): doar al doilea lot cu priorul initial Beta(2, 8) ---
alpha_post_c = alpha_prior + defecte_lot2
beta_post_c = beta_prior + (total_lot2 - defecte_lot2)
media_post_c = alpha_post_c / (alpha_post_c + beta_post_c)

print("\n=== SUBIECTUL 3 (c) ===")
print(f"Distributia a posteriori doar cu lotul 2: Beta({alpha_post_c}, {beta_post_c})")
print(f"Media a posteriori: {media_post_c:.6f} (exact: {alpha_post_c}/{alpha_post_c + beta_post_c})")
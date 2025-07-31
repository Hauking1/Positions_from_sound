import numpy as np
import matplotlib.pyplot as plt

# — Parameter
c, fs = 343, 44000
dx     = c / fs            # Distanz-Auflösung pro Sample
D      = np.sqrt(2)        # Abstand der Mics [m]

# — Funktionsdefinition
def spatial_resolution(theta):
    """
    Räumliche Auflösung für Ankunftswinkel theta (rad).
    Δs = dx / |sin(theta)|, mit clipping für theta≈0.
    """
    # Verhindere Division durch Null
    s = np.abs(np.sin(theta))
    s[s < 1e-3] = 1e-3   # untere Schranke, damit der Plot nicht explodiert
    return dx / s

# — Winkelachse von 0…180°
thetas = np.linspace(1e-3, np.pi-1e-3, 1000)
res    = spatial_resolution(thetas)

# — Plot
plt.figure(figsize=(6,4))
plt.plot(np.degrees(thetas), res, linewidth=1.2)
plt.xlabel('Ankunftswinkel θ [°]')
plt.ylabel('Räumliche Auflösung Δs [m]')
plt.title(f'Auflösung vs. Winkel (D={D:.3f} m, dx={dx:.2e} m)')
plt.axhline(dx, color='gray', linestyle='--', label=f'Best‑Case: Δs=dx≈{dx:.3f} m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# — Parameter
c, fs = 343, 44000
dx     = c / fs            # Distanz-Auflösung pro Sample ≈7.80 mm
D      = np.sqrt(2)        # Abstand der Mics [m] (hier nur dekorativ)

# — Funktion für die Abstandsauflösung
def radial_resolution(theta):
    """
    Radiale Auflösung entlang der Mikrophon-Achse:
    Δr = dx / |cos θ|, mit Clipping für θ≈90°.
    """
    cos_t = np.abs(np.cos(theta))
    # Vermeide Division durch Null
    cos_t[cos_t < 1e-3] = 1e-3
    return dx / cos_t

# — Winkelachse von 0…180°
thetas = np.linspace(0, np.pi, 1000)
res_r  = radial_resolution(thetas)

# — Plot
plt.figure(figsize=(6,4))
plt.plot(np.degrees(thetas), res_r, linewidth=1.2)
plt.xlabel('Ankunftswinkel θ [°]')
plt.ylabel('Radiale Auflösung Δr [m]')
plt.title(f'Abstandsauflösung vs. Winkel (dx={dx:.2e} m)')
plt.axhline(dx, color='gray', linestyle='--', label=f'Minimum Δr=dx≈{dx:.3f} m')
plt.ylim(0, np.max(res_r[np.degrees(thetas)<85]))  # Zoom bis 85°
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

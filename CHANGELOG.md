# Changelog

## [2.1.0] - 2026-02-06 : Stabilisation & Outils de Calibration

### Corrections & Fiabilisations
- Vision : passage à un mode **synchrone** (capture X11 + inférence YOLO) pour éviter les frames perdues en runtime.
- Correction du chargement YOLO : `torch.load` forcé avec `weights_only=False` pour éviter des erreurs d'unpickling sur certaines versions.
- Ajout d'un utilitaire `calibrate_roi.py` pour positionner la ROI et sauvegarder `GAME_WINDOW_X` / `GAME_WINDOW_Y`.
- Ajout du mode runtime d'ajustement d'offsets (F9 pour activer, F10 pour sauvegarder dans `config_hardware.py`).

### Améliorations
- Ajout de `test_capture.py` et `test_yolo_direct2.py` pour diagnostics rapides (capture et test d'inférence sur image).
- Réorganisation : déplacement des tests unitaires dans `test_unitaire/`.
- Ajustements de tuning : BEZIER_STEPS, SMOOTH_FACTOR, SENS_MULTIPLIER et micro-delays pour meilleur compromis réactivité/smoothness.

### Notes
- Garder `VisionSystem.py` séparé facilite le debug GPU/X11. Les autres modules utilitaires peuvent être fusionnés plus tard si besoin.

## [2.0.0] - 2026-02-05 : L'Algorithme "Human-Flow"

Cette version majeure introduit une refonte complète du moteur de visée. L'interpolation linéaire robotique est remplacée par une approche biomécanique basée sur les courbes de Bézier, rendant le mouvement indiscernable d'un joueur humain expérimenté.

<div align="center">
  <img src="Img_readme/Gif_fonctionnement.gif" alt="Démonstration Bézier G2" width="100%">
  <p><em>Ci-dessus : Tracking avec interpolation Bézier G2 et Head Offset dynamique.</em></p>
</div>

### 🚀 Nouveautés Majeures

#### 🌊 Interpolation Bézier G2 (Continuité de Courbure)
- Implémentation de courbes de Bézier quadratiques pour la trajectoire de la souris.
- **Inertie dynamique :** Le curseur préserve le "Momentum" du mouvement précédent, éliminant les changements de direction angulaires (robotiques) pour des courbes fluides et naturelles.
- Conformité G2 pour une invisibilité accrue face aux analyses heuristiques.

##### 🧠 Pourquoi la continuité G2 ? (Biomimétisme)

La plupart des aimbots utilisent une interpolation linéaire ou un lissage simple (**G1**), qui adoucit la trajectoire mais conserve des changements d'accélération brusques, invisibles à l'œil nu mais détectables par analyse heuristique.

Notre algorithme utilise des courbes de Bézier à **continuité G2** (Curvature Continuity).
* **G0 (Position) :** Le tracé est continu (pas de téléportation).
* **G1 (Tangente) :** La direction change sans angle vif (pas de saccade directionnelle).
* **G2 (Courbure) :** L'accélération elle-même est lissée. Il n'y a pas de "jerk" (à-coup) au début ou à la fin du mouvement.

**Résultat :** Le curseur se déplace avec une "inertie" simulée qui imite parfaitement la motricité fine des muscles de la main et du bras, rendant le tracking indiscernable d'un joueur humain de haut niveau.

#### 📉 Accélération "Ease-Out" Non-Linéaire
- Remplacement de la progression linéaire par une fonction racine carrée ($\sqrt{t}$).
- **Comportement :** Attaque rapide sur la cible (*Snap*) suivie d'une micro-décélération pour l'ajustement final, imitant la motricité fine humaine.

#### ⚡ Micro-Stepping Haute Fréquence
- Découpage de chaque instruction de mouvement en **3 micro-pas** interpolés.
- Permet de saturer le taux de rafraîchissement de la souris (polling rate) pour une fluidité absolue sur les moniteurs 180Hz+.

### ⚙️ Ajustements Techniques
- **Nouveau Moteur :** Intégration de la classe `BezierGenerator` pour la gestion des courbes G2.
- **Micro-Stepping 4X :** Augmentation de la résolution de mouvement à **4 micro-pas** par frame de détection. Cela maximise la fluidité sur les écrans haute fréquence en lissant davantage chaque trajectoire.
- **Réactivité :** `SMOOTH_FACTOR` fixé à **2.0**, offrant un équilibre agressif entre stabilité et vitesse d'acquisition.
- **Sensibilité :** `SENS_MULTIPLIER` ajusté à **1.65** pour garantir que le curseur couvre la distance nécessaire malgré l'amortissement de la courbe Bézier.
- **Précision :** `HEAD_OFFSET_PCT` calibré à **0.42** (niveau cou/menton) pour maximiser les chances de Headshot tout en restant sur la hitbox du corps.
- **Float Tracking :** Conservation des décimales (restes de pixels) pour une précision mathématique absolue sur la durée.
## [2.1.0] - 2026-02-06 : Stabilisation & Outils de Calibration

### Corrections & Fiabilisations
- Vision : passage à un mode **synchrone** (capture X11 + inférence YOLO) pour éviter les frames perdues en runtime.
- Correction du chargement YOLO : `torch.load` forcé avec `weights_only=False` pour éviter des erreurs d'unpickling sur certaines versions.
- Ajout d'un utilitaire `calibrate_roi.py` pour positionner la ROI et sauvegarder `GAME_WINDOW_X` / `GAME_WINDOW_Y`.
- Ajout du mode runtime d'ajustement d'offsets (F9 pour activer, F10 pour sauvegarder dans `config_hardware.py`).

### Améliorations
- Ajout de `test_capture.py` et `test_yolo_direct2.py` pour diagnostics rapides (capture et test d'inférence sur image).
- Réorganisation : déplacement des tests unitaires dans `test_unitaire/`.
- Petits ajustements de tuning : BEZIER_STEPS, SMOOTH_FACTOR, SENS_MULTIPLIER et micro-delays pour meilleur compromis réactivité/smoothness.

### Notes
- Garder `VisionSystem.py` séparé facilite le debug GPU/X11. Les autres modules utilitaires peuvent être fusionnés plus tard si besoin.

```
# Changelog

## [2.0.0] - 2026-02-05 : L'Algorithme "Human-Flow"

Cette version majeure introduit une refonte complète du moteur de visée. L'interpolation linéaire robotique est remplacée par une approche biomécanique basée sur les courbes de Bézier, rendant le mouvement indiscernable d'un joueur humain expérimenté.

<div align="center">
  <img src="Img_readme/Gif_fonctionnement.gif" alt="Démonstration Bézier G2" width="100%">
  <p><em>Ci-dessus : Tracking avec interpolation Bézier G2 et Head Offset dynamique.</em></p>
</div>

### 🚀 Nouveautés Majeures

#### 🌊 Interpolation Bézier G2 (Continuité de Courbure)
- Implémentation de courbes de Bézier quadratiques pour la trajectoire de la souris.
- **Inertie dynamique :** Le curseur préserve le "Momentum" du mouvement précédent, éliminant les changements de direction angulaires (robotiques) pour des courbes fluides et naturelles.
- Conformité G2 pour une invisibilité accrue face aux analyses heuristiques.

##### 🧠 Pourquoi la continuité G2 ? (Biomimétisme)

La plupart des aimbots utilisent une interpolation linéaire ou un lissage simple (**G1**), qui adoucit la trajectoire mais conserve des changements d'accélération brusques, invisibles à l'œil nu mais détectables par analyse heuristique.

Notre algorithme utilise des courbes de Bézier à **continuité G2** (Curvature Continuity).
* **G0 (Position) :** Le tracé est continu (pas de téléportation).
* **G1 (Tangente) :** La direction change sans angle vif (pas de saccade directionnelle).
* **G2 (Courbure) :** L'accélération elle-même est lissée. Il n'y a pas de "jerk" (à-coup) au début ou à la fin du mouvement.

**Résultat :** Le curseur se déplace avec une "inertie" simulée qui imite parfaitement la motricité fine des muscles de la main et du bras, rendant le tracking indiscernable d'un joueur humain de haut niveau.

#### 📉 Accélération "Ease-Out" Non-Linéaire
- Remplacement de la progression linéaire par une fonction racine carrée ($\sqrt{t}$).
- **Comportement :** Attaque rapide sur la cible (*Snap*) suivie d'une micro-décélération pour l'ajustement final, imitant la motricité fine humaine.

#### ⚡ Micro-Stepping Haute Fréquence
- Découpage de chaque instruction de mouvement en **3 micro-pas** interpolés.
- Permet de saturer le taux de rafraîchissement de la souris (polling rate) pour une fluidité absolue sur les moniteurs 180Hz+.

### ⚙️ Ajustements Techniques
- **Nouveau Moteur :** Intégration de la classe `BezierGenerator` pour la gestion des courbes G2.
- **Micro-Stepping 4X :** Augmentation de la résolution de mouvement à **4 micro-pas** par frame de détection. Cela maximise la fluidité sur les écrans haute fréquence en lissant davantage chaque trajectoire.
- **Réactivité :** `SMOOTH_FACTOR` fixé à **2.0**, offrant un équilibre agressif entre stabilité et vitesse d'acquisition.
- **Sensibilité :** `SENS_MULTIPLIER` ajusté à **1.65** pour garantir que le curseur couvre la distance nécessaire malgré l'amortissement de la courbe Bézier.
- **Précision :** `HEAD_OFFSET_PCT` calibré à **0.42** (niveau cou/menton) pour maximiser les chances de Headshot tout en restant sur la hitbox du corps.
- **Float Tracking :** Conservation des décimales (restes de pixels) pour une précision mathématique absolue sur la durée.
# Changelog - Projet Aimbot YOLO

## [1.5.0] - 2026-02-04
### Ajouté
- Détection spécifique des modèles humains (CS2/Aimlab).
- Calcul dynamique du point de visée (Head Offset à 45%).
- Nouveau filtre de micro-mouvements pour stabiliser le viseur.
- Touche [Page Down] pour le monitoring en temps réel.

### Changé
- Sensibilité calibrée pour 800 DPI / 1.92 in-game.
- Augmentation de la résolution d'analyse (imgsz=640).
- Lissage du mouvement (Smooth Factor 3.2).

### Corrigé
- Bug de variable 'SENS_MULTIPLIER' non définie.
- Latence d'accumulation des frames dans la queue de capture.
# 🔭 Visée Optronique (Autonomous Optronic Sighting)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![YOLO](https://img.shields.io/badge/AI-YOLOv10-orange)
![Platform](https://img.shields.io/badge/Platform-Linux%20Pop!_OS-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Visée Optronique** est un système expérimental de vision par ordinateur temps réel conçu pour l'acquisition et le suivi de cibles dans des environnements de simulation rapide (ex: Aim Lab).

Développé nativement pour l'architecture **Linux (X11)**, ce projet explore les limites de la latence dans la boucle de détection-action en couplant l'inférence neuronale (YOLO) avec l'injection d'entrées au niveau du noyau (uinput).

---

## ⚡ Caractéristiques Techniques

* **Moteur de Vision Neurale** : Utilisation de **YOLOv10** optimisé pour l'inférence ultra-rapide (<10ms) sur GPU NVIDIA (RTX Series).
* **Pipeline de Capture Hybride** : Capture d'écran haute fréquence via `mss` avec gestion dynamique des *Regions of Interest* (ROI) pour minimiser la charge CPU.
* **Contrôle Moteur Humanoïde** :
    * Algorithme de lissage dynamique (Exponential Smoothing) pour éviter les mouvements robotiques.
    * Trajectoires de Bézier non-linéaires.
    * Correction de trajectoire en temps réel (boucle fermée 60Hz).
* **Intégration Système** : Injection d'événements souris via le module noyau `uinput` pour une émulation matérielle indétectable logiciellement.

## 🛠️ Architecture

Le système fonctionne en boucle fermée selon le schéma suivant :

1.  **Acquisition** : Capture brute du framebuffer (Xorg/X11).
2.  **Inférence** : Détection des cibles via Réseau de Neurones Convolutifs (CNN).
3.  **Décision** : Calcul vectoriel de la cible prioritaire (plus proche voisin).
4.  **Action** : Calcul du delta de mouvement et lissage PID.

## 📋 Prérequis

* **OS** : Linux (Testé sur Pop!_OS 22.04 LTS).
* **Serveur d'affichage** : **X11 / Xorg** (Wayland n'est pas supporté pour la capture haute performance).
* **GPU** : NVIDIA recommandé (CUDA supporté).
* **Dépendances** : Python 3.10+, `uinput`, `mss`, `ultralytics`, `opencv-python`.

## 🚀 Installation

1. **Cloner le dépôt**
   ```bash
   git clone [https://github.com/ton-pseudo/visee-optronique.git](https://github.com/ton-pseudo/visee-optronique.git)
   cd visee-optronique
"""
Configuration optimisée pour:
- AMD Ryzen 9 9600X (6 cores / 12 threads)
- 32GB RAM
- RTX 5070 (12GB VRAM)

Ce fichier contient les réglages pour exploiter pleinement votre hardware
"""

# =============================================================================
# OPTIMISATIONS GPU (RTX 5070)
# =============================================================================

# Utiliser TensorRT pour gains massifs (2-3x plus rapide)
USE_TENSORRT = True

# Taille de workspace VRAM pour TensorRT (en GB)
TENSORRT_WORKSPACE = 4  # Sur 12GB disponibles

# Confidence threshold (GPU puissant = peut gérer plus de détections)
# BAISSÉ à 0.25 pour FORCER les détections (test)
YOLO_CONFIDENCE = 0.25

# Détections max par frame (limiter post-processing)
YOLO_MAX_DETECTIONS = 10

# Filtre Y pour éliminer ciel (en pixels depuis le haut)
# ENLEVÉ pour test (MIN_Y_THRESHOLD = 0)
MIN_Y_THRESHOLD = 0

# =============================================================================
# CALIBRAGE DE LA FENÊTRE DE JEU (Multi-moniteur)
# =============================================================================

# Coin haut-gauche de la fenêtre de jeu (calibré manuellement pour ton setup)
GAME_WINDOW_X = 1074
GAME_WINDOW_Y = 1420

# Offset fin pour corriger le décalage du viseur (en pixels dans la ROI)
# Valeurs positives: droite/bas | négatives: gauche/haut
AIM_OFFSET_X = -2
AIM_OFFSET_Y = 1

# =============================================================================
# OPTIMISATIONS RAM (32GB)
# =============================================================================

# Taille des buffers de queue (profiter de la RAM)
FRAME_QUEUE_SIZE = 3      # Au lieu de 1
DETECTION_QUEUE_SIZE = 2  # Au lieu de 1

# =============================================================================
# OPTIMISATIONS CPU (9600X - 12 threads)
# =============================================================================

# Nombre de threads pour capture/traitement
NUM_CAPTURE_THREADS = 2   # Multi-threaded capture possible
NUM_OPENCV_THREADS = 4    # OpenCV peut utiliser plus de threads

# =============================================================================
# TRACKING & MOUVEMENT (Adaptés au hardware rapide)
# =============================================================================

# Avec hardware puissant, on peut augmenter la précision
BEZIER_STEPS = 16          # Plus de micro-steps = moins saccadé
KALMAN_PREDICTION_FRAMES = 5  # Prédire jusqu'à 5 frames dans le futur

# Smooth adaptatif (ajusté pour haute réactivité)
SMOOTH_FACTOR = 1.6        # Milieu entre smooth et réactif
SENS_MULTIPLIER = 2.2      # Vitesse intermédiaire

# Délai entre micro-mouvements (hardware rapide = moins de délai)
MICRO_MOVEMENT_DELAY = 0.00005  # Encore plus fluide

# =============================================================================
# PARAMÈTRES AVANCÉS
# =============================================================================

# Activer Mixed Precision (RTX)
USE_MIXED_PRECISION = True

# CUDA optimizations
CUDA_BENCHMARK = True
CUDA_TF32 = True

# Préchargement du modèle
WARMUP_ITERATIONS = 10

print("""
╔════════════════════════════════════════════════════════════╗
║  CONFIGURATION OPTIMISÉE HARDWARE                          ║
║  • RTX 5070 (12GB) : TensorRT + Mixed Precision           ║
║  • 32GB RAM        : Buffers augmentés (3x)               ║
║  • 9600X (12T)     : Multi-threading optimisé             ║
║                                                            ║
║  Gains attendus: 2-4x plus rapide                         ║
╚════════════════════════════════════════════════════════════╝
""")

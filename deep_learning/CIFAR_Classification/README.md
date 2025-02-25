# Deep-Learning
Deep Learning

CNN Classification sur CIFAR-10

Ce projet implémente un réseau de neurones convolutionnel (CNN) en PyTorch pour classifier les images du dataset CIFAR-10.

📌 Description

Le modèle utilise deux couches convolutionnelles, suivies de couches entièrement connectées, et est entraîné à l'aide de la fonction de perte CrossEntropyLoss et de l'optimiseur SGD.

📂 Contenu du script

Chargement et transformation du dataset CIFAR-10

Définition du CNN

Entraînement du modèle

Évaluation des performances sur le jeu de test

🛠️ Installation

Cloner le dépôt
```bash
git clone https://github.com/dsteve87/AI-projects.git
```
Installer les dépendances

```bash
pip install torch torchvision matplotlib
```

Lancer l'entraînement
```bash
python cifar10_cnn.py
```

🔍 Résultats

Le modèle est évalué sur le jeu de test et affiche l'accuracy globale ainsi que l'accuracy par classe.

📜 Remarques

GPU supporté : Si une GPU CUDA est disponible, le modèle l'utilisera automatiquement.

Hyperparamètres modifiables dans le script (nombre d'époques, learning rate, etc.).

📖 Références

PyTorch Documentation

CIFAR-10 Dataset

🚀 Bon entraînement !

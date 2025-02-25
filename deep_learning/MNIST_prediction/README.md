# Deep-Learning


## MLP Classification sur MNIST

Ce projet implémente un réseau de neurones multicouches (MLP) en PyTorch pour classifier les images du dataset MNIST.

📌 **Description**

Le modèle utilise une architecture simple avec une couche cachée de 500 neurones, suivie d'une couche de sortie pour prédire les chiffres de 0 à 9. Il est entraîné à l'aide de la fonction de perte **CrossEntropyLoss** et de l'optimiseur **Adam**.

📂 **Contenu du script**

- Chargement et transformation du dataset MNIST
- Définition du modèle MLP
- Entraînement du modèle
- Évaluation des performances sur le jeu de test

🛠️ **Installation**

1. **Cloner le dépôt** :
    ```bash
    git clone https://github.com/dsteve87/AI-projects.git
  
    ```

2. **Installer les dépendances** :
    ```bash
    pip install torch torchvision matplotlib
    ```

3. **Lancer l'entraînement** :
    ```bash
    python MLP_MNIST_prediction.py
    ```

🔍 **Résultats**

Le modèle est évalué sur le jeu de test et affiche l'accuracy globale du modèle (96.84 % sur deux époques).

📜 **Remarques**

- **GPU supporté** : Si un GPU CUDA est disponible, le modèle l'utilisera automatiquement.
- **Hyperparamètres modifiables** dans le script : nombre d'époques, taux d'apprentissage, taille des lots, etc.

📖 **Références**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

🚀 **Bon entraînement !**


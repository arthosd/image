# image

Ce projet vise à determiner le niveau de remplissage d'un verre rempli d'un liquide quelconque.

## Getting started

Avant de compiler le projet vérifier que OPENCV est bien installé sur votre machine. Quand cela est fait, renseignez le chemin vers lequel pointe l'installation d'openCV dans le fichier CMakeLists.txt .

La configuration d'openCV étant terminé, effectuez les commandes suivantes (à partir de la racine du projet) :

Il faut que CMAKE soit installé sur votre machine pour procéder à la compilation du projet.

- ### Installation Linux ou Mac

```bash
mkdir build
cd build/
cmake ...
make
```

- ### Installation sur Windows

  Aller à la racine du projet et créer le dossier build. Excecutez le fichier CMakeLists.txt à l'aide du programme CMake.  
  A l'aide d'un make de votre choix, excecutez le makefile généré.

  ## Running

  Après avoir compilé le projet vous pouvez exécuter le fichier Detector.

- Linux / Mac

  ```bash
  ./Detector
  ```

- Windows
  Double cliquez sur l'excecutable généré.

## Note

**_Il y a des erreurs qui peuvent apparaitre durant l'excécution du projet (des segmentations faults). Ces erreures sont dû à la fonction HoughLines d'openCV._**

Si cette erreure intervient durant une excecution, relancer le programme.

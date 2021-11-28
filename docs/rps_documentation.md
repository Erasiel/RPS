# Kő-papír-olló

## Setup

A program indításához szükséges függőségek (verziószám, ahol fontos):
- Python 3.7+
- Python csomagok:
  - tensorflow
  - keras
  - mediapipe
  - python-opencv
  - scikit-learn 1.0.1+
  - PyQt5

Eszköz követelmények:
- Webkamera (min. 256x256 pixeles felbontással)

A program indítása: `python App.py`

## Működés

A játékos a választott kézállást mutassa a kamera képernyőn is látszó részébe. Egyjátékos mód esetén ez a kamera középső 256x256 pixeles része, kétjátékos mód esetén a jobb-, illetve baloldali legszélső (függőlegesen középső) 256x256 pixeles rész. **A játékterület legyen jól megvilágított.** 

Amikor a játékos(ok) kézállása végleges, a szoftver kezelője kattintson a `Capture` feliratú gombra. Ennek hatására egyjátékos módban lefut a kézállások detektálása, ezek után egyjátékos módban a gép is választ egy kézállást (ez nem feltétlenül függ a játékos aktuálisan választott kézállásától), legvégül a játék eredményt hirdet.

Játékmódok váltására a bal oldali gombokkal van lehetőség. A játékos három különbüző nehézségű egyjátékos mód és a többjátékos mód közül választhat. A fekete színnel megjelenő gomb jelzi az aktív nehézséget.

A kézállások detektálásának módszerét is lehet változtatni a felső két gombbal, ahol szintén a fekete színnel megjelenő gomb jelzi az aktív detektálási módszert.

### Választható játékmódok

Egyjátékos mód:
- Könnyű: a gép véletlenszerűen választ (a játékos aktuális választásától függetlenül)
- Normál: a gép a játékos utolsó két lépése alapján statisztikai alapon választ (a játékos aktuális választásától függetlenül)
- Nehéz: a gép úgy választ, hogy legyőzze a játékost (a játékos aktuális választása alapján)

Többjátékos mód:
- A játékosok egymás ellen játszanak egy kamera képén keresztül. 

### Választható detektálási módszerek

- Neuronhálós detektálás: a pozíciót egy neurális háló osztályozza
- Mediapipe-alapú detektálás: a pozíció meghatározása a mediapipe keretrendszer segítségével kigyűjtött jellemzők alapján kerül meghatározásra

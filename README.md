# DOCUMENTATIE PROIECT MLP - CLASIFICARE MULTI-CLASS

## 1. Descrierea Modelului
### Preprocesarea Datelor
Datele de intrare provin din doua fisiere CSV: train_samples.csv (caracteristici) si train_labels.csv (etichete). Deoarece dimensiunile nu corespund, se aplica transpusa matricei de caracteristici pentru a obtine forma corecta.
**Reducerea dimensionalitatii (PCA):** Se aplica PCA pentru a reduce numarul de caracteristici de la 784 la 80 componente. Dupa testarea cu diferite numere de componente, 80 a avut cele mai bune rezultate. PCA este antrenat doar pe datele de antrenare din fiecare fold, iar transformarea este aplicata si pe datele de testare pentru a evita data leakage. In acest mod, pastram cele mai importante caracteristici si acceleram antrenarea retelei.
**Etichete:** Clasele originale 1-10 sunt convertite la indexare 0-based(0-9) pentru compatibilitate cu functia de pierdere sparse_categorical_crossentropy.

### Arhitectura Retelei Neurale (MLP)
Modelul este un Multilayer Perceptron implementat cu tensorflow.keras, cu urmatoarea structura:
| Strat         | Tip               | Neuroni / Parametri | Activare |
|--------------|------------------|---------------------|----------|
| Input        | Input Layer      | 80 (PCA components) | —        |
| Strat 1      | Dense            | 256 neuroni         | ReLU     |
| Batch Norm 1 | BatchNormalization | —                 | —        |
| Dropout 1    | Dropout          | p = 0.4             | —        |
| Strat 2      | Dense            | 128 neuroni         | ReLU     |
| Batch Norm 2 | BatchNormalization | —                 | —        |
| Dropout 2    | Dropout          | p = 0.3             | —        |
| Strat 3      | Dense            | 32 neuroni          | ReLU     |
| Output       | Dense            | 10 neuroni          | Softmax  |

Functia de activare ReLu a fost aleasa deoarece este cea mai buna functie la momentul de fata, deoarece elimina problema vanishing gradient si converge mai rapid fata de sigmoid sau tanh (desi datele noastre sunt intre 0 si 1). BatchNormalization normalizeaza activarile intre straturi, imbunatatind stabilitatea si viteza de antrenare.
Straturile de Dropout previn overfitting-ul prin dezactivarea aleatorie a neuronilor in faza de antrenare.
Functia de activare Softmax a fost folosita in ultimul strat cu 10 neuroni (corespunde cu numarul de clase) deoarece alege clasa cu cea mai mare probabilitate (suma totala = 1).

### Algoritmul de Antrenare

| Parametru                     | Valoare / Descriere                                              |
|------------------------------|------------------------------------------------------------------|
| Optimizator                  | Adam (Adaptive Moment Estimation)                                |
| Functia de pierdere          | sparse_categorical_crossentropy                                  |
| Metrica de evaluare          | Accuracy                                                         |
| Numar maxim de epoci         | 150                                                              |
| Batch size                   | 16                                                               |
| Validation split             | 10% din datele de antrenare                                      |
| Early Stopping (monitor)     | val_loss                                                         |
| Early Stopping (patience)    | 5 epoci                                                          |
| Early Stopping (restore_best)| True — se restaureaza ponderile de la epoca cu cel mai mic val_loss |

Ca si optimizator, am incercat Adam si SGD cu diferiti parametrii, dar diferenta dintre ele a fost extrem de mica, iar Adam default a avut cel mai bun rezultat. Functia de pierdere sparse_category_crossentropy este potrivita pentru clasificare multi-class cu etichete intregi. Early stopping opreste antrenarea atunci cand performanta pe setul de validare nu se imbunatateste timp de 5 epoci consecutive, prevenind overfitting-ul si reducand timpul de antrenare. Batch size = 16 a avut cel mai bun rezultat. 

### Modelul Final
Dupa cross-validation, modelul final este antrenat pe intregul set de date de antrenare folosind aceeasi arhitectura. Numarul de epoci este stabilit 17 dupa ce am facut mediana fold-urilor. PCA este re-antrenat pe toate datele inainte de antrenarea modelului final.

Evolutia antrenarii modelului final:

| Epocă | Steps | Accuracy | Loss   |
|------|-------|----------|--------|
| 1    | 63/63 | 0.3294   | 2.1060 |
| 2    | 63/63 | 0.7413   | 0.8440 |
| 3    | 63/63 | 0.8491   | 0.5232 |
| 4    | 63/63 | 0.8707   | 0.4313 |
| 5    | 63/63 | 0.9005   | 0.3367 |
| 6    | 63/63 | 0.8894   | 0.3457 |
| 7    | 63/63 | 0.9276   | 0.2371 |
| 8    | 63/63 | 0.9235   | 0.2468 |
| 9    | 63/63 | 0.9428   | 0.1948 |
| 10   | 63/63 | 0.9398   | 0.1605 |
| 11   | 63/63 | 0.9326   | 0.2014 |
| 12   | 63/63 | 0.9538   | 0.1574 |
| 13   | 63/63 | 0.9646   | 0.1328 |
| 14   | 63/63 | 0.9757   | 0.0934 |
| 15   | 63/63 | 0.9658   | 0.1143 |
| 16   | 63/63 | 0.9616   | 0.1191 |
| 17   | 63/63 | 0.9740   | 0.1032 |

Modelul final atinge o acuratete de 97,40% pe setul complet de antrenare la epoca 17, cu o pierdere de 0.1032, demonstrand o convergenta buna fara semne de overfitting.

## 2. Rezultatele Validarii Incrucisate 10-fold

### Acuratetea pe fiecare fold

| Fold | Acuratete | Epoca de oprire |
|------|-----------|---------------------|
| 1    | 0.9300 | 19   |
| 2    | 0.9300 | 15  |
| 3    | 0.9600 | 16   |
| 4    | 0.8900 | 17   |
| 5    | 0.8900 | 20   | 
| 6    | 0.9400 | 17  | 
| 7    | 0.8900 | 33   | 
| 8    | 0.9400 | 16  |
| 9    | 0.9500 | 23   | 
| 10   | 0.9800 | 17  | 
| MEDIE   | 0.9300 | 17(mediana)   |

### Interval de Incredere 90%

| Metrica | Valoare | 
|---------|-----------|
| Acuratete medie |0.9300 | 
| Deviatie standard    | 0.0297 | 
| Limita inf. CI 90%    | 0.9146 | 
| Limita sup. CI 90%    | 0.9454 |
| Interval de incredere 90%   | (0.9146, 0.9454) |  

Intervalul (0.9146, 0.9454) indica faptul ca, cu o probabilitate de 90%, acuratetea reala a modelului pe date nevazute se afla in acest interval.

## Concluzie
Modelul implementat obtine o acuratete medie de 93% pe setul de antrenare evaluat prin 10-fold cross validation. Aceste rezultate demonstreaza o performanta destul de ridicata a modelului, cu doar 1000 de samples. 


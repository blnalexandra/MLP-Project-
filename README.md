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

Functia de activare ReLu a fost aleasa deoarece elimina problema vanishing gradient si converge mai rapid fata de sigmoid sau tanh. BatchNormalization normalizeaza activarile intre straturi, imbunatatind stabilitatea si viteza de antrenare.
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

Ca si optimizator, am incercat Adam si SGD cu diferiti parametrii, dar diferenta dintre ele a fost extrem de mica, iar Adam default a avut cel mai bun rezultat. Functia de pierdere sparse_category_crossentropy este potrivita pentru clasificare multi-class cu etichete intregi. Early stopping opreste antrenarea atunci cand performanta pe setul de validare nu se imbunatateste timp de 10 epoci consecutive, prevenind overfitting-ul si reducand timpul de antrenare. Batch size = 16 a avut cel mai bun rezultat. 

### Modelul Final
Dupa cross-validation, modelul final este antrenat pe intregul set de date de antrenare folosind aceeasi arhitectura. Numarul de epoci este stabilit 27 dupa ce am facut mediana fold-urilor. PCA este re-antrenat pe toate datele inainte de antrenarea modelului final.

Evolutia antrenarii modelului final:

| Epoca | Accuracy | Loss   |
| ----- | -------- | ------ |
| 1     | 0.2716   | 2.1584 |
| 2     | 0.6889   | 1.1300 |
| 3     | 0.7737   | 0.7688 |
| 4     | 0.8562   | 0.5579 |
| 5     | 0.8671   | 0.4192 |
| 6     | 0.8949   | 0.3041 |
| 7     | 0.9183   | 0.3139 |
| 8     | 0.9314   | 0.2334 |
| 9     | 0.9375   | 0.2128 |
| 10    | 0.9296   | 0.2197 |
| 11    | 0.9376   | 0.1993 |
| 12    | 0.9364   | 0.2053 |
| 13    | 0.9369   | 0.1950 |
| 14    | 0.9607   | 0.1405 |
| 15    | 0.9508   | 0.1447 |
| 16    | 0.9475   | 0.1735 |
| 17    | 0.9601   | 0.1268 |
| 18    | 0.9472   | 0.1529 |
| 19    | 0.9399   | 0.1476 |
| 20    | 0.9530   | 0.1309 |
| 21    | 0.9499   | 0.1547 |
| 22    | 0.9686   | 0.1025 |
| 23    | 0.9660   | 0.1017 |
| 24    | 0.9660   | 0.1078 |
| 25    | 0.9652   | 0.1055 |
| 26    | 0.9717   | 0.0885 |
| 27    | 0.9812   | 0.0693 |


Modelul final atinge o acuratete de 98,12% pe setul complet de antrenare la epoca 27, cu o pierdere de 0.0693, demonstrand o convergenta buna.

## 2. Rezultatele Validarii Incrucisate 10-fold

### Acuratetea pe fiecare fold

| Fold | Acuratețe | Epoca de oprire |
| ---- | --------- | --------------- |
| 1    | 0.9300    | 27              |
| 2    | 0.9500    | 35              |
| 3    | 0.9700    | 28              |
| 4    | 0.8800    | 33              |
| 5    | 0.8800    | 21              |
| 6    | 0.9600    | 26              |
| 7    | 0.8900    | 34              |
| 8    | 0.9600    | 22              |
| 9    | 0.9500    | 23              |
| 10   | 0.9600    | 29              |
| MEDIE   | 0.9330 | 27(mediana)   |

### Interval de Incredere 90%

| Metrica | Valoare | 
|---------|-----------|
| Acuratete medie |0.9330 | 
| Deviatie standard    | 0.0341 | 
| Limita inf. CI 90%    | 0.9153 | 
| Limita sup. CI 90%    | 0.9507 |
| Interval de incredere 90%   | (0.9153, 0.9507) |  

Intervalul (0.9153, 0.9507) indica faptul ca, cu o probabilitate de 90%, acuratetea reala a modelului pe date nevazute se afla in acest interval.

## Concluzie
Modelul implementat obtine o acuratete medie de 93% pe setul de antrenare evaluat prin 10-fold cross validation. Aceste rezultate demonstreaza o performanta destul de ridicata a modelului, cu doar 1000 de samples. 


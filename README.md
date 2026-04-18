# DOCUMENTATIE PROIECT MLP - CLASIFICARE MULTI-CLASS

## 1. Descrierea Modelului
### Preprocesarea Datelor
Datele de intrare provin din doua fisiere CSV: train_samples.csv (caracteristici) si train_labels.csv (etichete). Deoarece dimensiunile nu corespund, se aplica transpusa matricei de caracteristici pentru a obtine forma corecta.
**Reducerea dimensionalitatii (PCA):** Se aplica PCA pentru a reduce numarul de caracteristici de la 784 la 80 componente. PCA este antrenat doar pe datele de antrenare din fiecare fold, iar transformarea este aplicata si pe datele de testare pentru a evita data leakage. In acest mod, pastram cele mai importante caracteristici si acceleram antrenarea retelei.
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




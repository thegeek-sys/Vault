---
Created: 2024-12-13
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Quando le chiavi ammettono un ordinamento significativo per l’applicazione, è più conveniente utilizzare un’organizzazione fisica dei fati che ne tanga conto (nel caso di campi multipli, si ordina sul primo campo, poi sul secondo e così via)

---
## File ISAM
Il file **ISAM** (Indexed Sequential Access Method) è il primo esempio significativo della richiesta fatta.

Il file principale viene ordinato in base al valore della chiave di ricerca e generalmente viene lasciata una carta percentuale di spazio libero in ogni blocco (necessario per l’inserimento)

![[Pasted image 20241213170716.png|500]]
Nel file ISAM si ha un **file indice** (table of contents) e un **file principale**.
All’interno del file indice è presente una entrata per ogni blocco del file principale composta da chiave e puntatore all’inizio del blocco. La chiave in questo caso corrisponde alla chiave del primo record di ogni blocco del file principale (fatta eccezione per la prima entrata che è $-\infty$).
Ogni record del file indice inoltre è $\leq$ delle chiavi del blocco puntato ma strettamente maggiore delle chiavi del blocco puntato dal record precedente

---
## Ricerca
Per ricercare un record con valore della chiave $k$ occorre ricercare sul file indice un valore $k'$ della chiave che **ricopre $k$**, cioè tale che:
- $k'\leq k$
- se il record con chiave $k'$ non è l’ultimo record del file indice e $k''$ è il valore della chiave nel record successivo $k<k''$

La ricerca di un record con chiave $k$ richiede una **ricerca sul file indice** + **1 accesso** in lettura sul file principale (nel conto del costo dobbiamo considerare solamente il numero di accessi in memoria, non importa la ricerca su un file che è stato già portato in memoria principale)

Poiché il file indice è ordinato in base al valore della chiave, la ricerca di un valore che ricopre la chiave può essere fatta in modo efficiente mediante la **ricerca binaria**

> se $k>k_{1}$ allora verifico tutte le chiavi successive all’interno del blocco e controllo se trovo $k_{n}$ tale che $k_{n}<k<k_{n+1}$ in tal caso la ricerca dovrà proseguire nel blocco puntato da $k_{n}$ (questa ricerca non aggiunge costo, sto leggendo un blocco già caricato); in caso contrario ($k$ maggiore di tutte le chiavi del blocco) prosegue la ricerca binaria sui blocchi da 

### Ricerca binaria
Si fa un accesso in lettura al blocco del file indice $\frac{m}{2}+1$ e si confronta $k$ con $k_{1}$ (prima chiave del blocco):
- se $k=k_{1}$ abbiamo finito
- se $k<k_{1}$ allora si ripete il procedimento sui blocchi da $1$ a $\frac{m}{2}$
- se $k>k_{1}$ allora la ricerca prosegue sui blocchi da $\frac{m}{2}+1$ ad $m$ (il blocco $\frac{m}{2}+1$ va riconsiderato perché abbiamo controllato solo la prima chiave)

Ci si ferma quando lo spazio di ricerca è ridotto ad un solo blocco, quindi dopo $\boldsymbol{\lceil \log_{2}m \rceil}$ accessi

### Ricerca per interpolazione
La **ricerca per interpolazione** è basata sulla conoscenza della **distribuzione** dei valori della chiave, ovvero deve essere disponibile una funzione $f$ che dati tre valori $k_{1}$, $k_{2}$, $k_{3}$ della chiave fornisce un valore che è la **frazione dell’intervallo di valori** dalla chiave compresa tra $k_{2}$ e $k_{3}$ in cui deve trovarsi $k_{1}$ cioè la chiave che stiamo cercando (nella ricerca binaria questa frazione è sempre $\frac{1}{2}$). Ad esempio quando cerchiamo in un dizionario non partiamo sempre da metà

$k_{1}$ deve essere confrontato con il valore $k$ della chiave del primo record del blocco $i$ (del file indice), dove $i=f(k_{1},k_{2},k_{3})\cdot m$; analogamente a quanto accade nella ricerca binaria. Se $k_{1}$ è minore di tale valore 


---
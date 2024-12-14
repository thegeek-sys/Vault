---
Created: 2024-12-14
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Il fatto di avere un ordinamento delle chiavi ci ha dato buoni risultati ed ha migliorato le prestazioni in termini di numero di accessi a memoria

Il **b-tree** nasce dalla **generalizzazione della struttura di indice** (estensione del file ISAM). Si accede al file attraverso una gerarchia di indici: l’indice al livello più alto nella gerarchia (la **radice**) è costituito da un un **unico blocco** e quindi può risiedere in memoria principale durante l’utilizzo del file
Ogni blocco di un file indice è costituito di record contenenti una coppia $(v,b)$ dove $v$ è il valore della chiave del primo record della porzione del file principale che è accessibile attraverso il puntatore $b$; $b$ può essere un puntatore ad un blocco del file indice a livello immediatamente più basso oppure, nel caso sia il livello più basso del file indice, ad un blocco principale

Il primo record indice di ogni blocco contiene solo un puntatore ad un blocco le cui chiavi sono minori del blocco puntato dal secondo record indice.
Un blocco del file indice è memorizzato come quello di un ISAM; ma ogni blocco di un b-tree (indice o file principale) deve essere **pieno almeno per metà** della sua dimensione tranne eventualmente la radice (basta che contenga almeno due entrate)

![[Pasted image 20241214161857.png]]

>[!info]
>Ogni record indice ha una chiave che ricopre quelle del sottoalbero che parte dal blocco puntato

---
## Ricerca
Durante la ricerca di un record con un dato valore per la chiave si accede agli indice a partire da quello a livello più alto; a mano a mano che si scende nella gerarchia di indice si restringe la porzione (insieme di blocchi) del file principale in cui deve trovarsi il record desiderato, fino a che, nell’ultimo livello (più basso nella gerarchia) tale porzione è ristretta ad un unico blocco

Più in dettaglio, per ricercare il record del file principale con un dato valore $v$ per la chiave si procede nel modo seguente. Si parte dall’indice a livello più alto (che è costituito da un unico blocco) e ad ogni passo si esamina un unico blocco. Se il blocco esaminato è un blocco del file principale, tale blocco è quello in deve trovarsi il record desiderato; se, invece, è un blocco di un file indice, si cerca in tale blocco un valore della chiave che ricopre $v$ e si segue il puntatore associato (che sarà o un puntatore ad un blocco dell’indice al livello immediatamente inferiore o un blocco del file principale)

![[Pasted image 20241214164234.png]]

Per la ricerca sono necessari $\boldsymbol{h+1}$ accessi dove $h$ è l’altezza dell’albero (basta visitare un blocco per ogni livello)

>[!info] Osservazione
>L’altezza dell’albero dipende da quanto sono pieni i blocchi (dalla loro densità). Infatti più sono i blocchi più $h$ è piccolo (e quindi meno costa la ricerca)
>Come conseguenza di ciò si richiede che ogni blocco (sia del file principale sia dell’indice) sia pieno almeno per metà
>
>Se i blocchi sono completamente pieni un inserimento può richiedere una modifica dell’indice ad ogni livello e in ultima ipotesi può far crescere l’altezza dell’albero di un livello

---
## Massimo valore di $h$
Siano:
- $N$ numero di record del file principale
- $2e-1$ numero massimo di record del file principale che possono essere memorizzati in un blocco ($e$  numero minimo di record per ogni foglia)
- $2d-1$ numero massimo di record del file indice che possono essere memorizzati in un blocco ($d$ numero minimo di record per ogni blocco indice)

>[!info]
>L’assunzione che il numero di record del file principale e del file indice che possono essere memorizzati in un blocco sia dispari viene fatta esclusivamente per rendere semplici i calcoli

L’altezza massima dell’albero denotata con $k$ si ha quando i blocchi sono pieni al minimo, cioè quando:
- ogni blocco del file principale contiene esattamente $e$ record
- ogni blocco del file indice contiene esattamente $d$ record
Pertanto:
- il file principale ha al più $\frac{N}{e}$ blocchi
- al livello $1$ il file indice ha $\frac{N}{e}$ record che possono essere memorizzati in $\frac{N}{ed}$ blocchi
- al livello $2$ il file indice ha $\frac{N}{ed}$ record che possono essere memorizzati in $\frac{N}{ed^2}$ blocchi
- …
- al livello $i$ il file indice ha $\frac{N}{ed^{i-1}}$ record che possono essere memorizzati in $\frac{N}{ed^i}$ blocchi

Al livello $k$ il file indice ha esattamente $1$ blocco quindi $\left\lceil  \frac{N}{ed^k}  \right\rceil=1$. Consideriamo per semplicità che $\frac{N}{ed^k}=1$ da cui $d^k=\frac{N}{e}$ e infine $k=\log_{g}\left( \frac{N}{e} \right)$

Quindi il valore che rappresenta il limite superiore per l’altezza dell’albero è
$$
k=\log_{d}\left( \frac{N}{e} \right)
$$

---
## Inserimento
Il costo di un inserimento **se nel blocco c’è spazio sufficiente** per inserire il record è il costo di una ricerca per trovare il blocco in cui deve essere inserito il record più un accesso per riscrivere il blocco $h+1+1$

Il costo di un inserimento **se nel blocco non c’è spazio sufficiente** per inserire il record è il costo di una ricerca per trovare il blocco in cui deve essere inserito il record più $s\leq 2h+1$ (nel caso peggiore per ogni livello dobbiamo sdoppiare un blocco quindi effettuare due accessi più uno alla fine per la nuova radice)

### Esempi
>[!example]- C’è spazio sufficiente
>Immaginiamo di voler inserire il record con chiave $22$
>
>![[Pasted image 20241214174528.png]]
>
>Inserendolo si ha quindi
>![[Pasted image 20241214174558.png]]

>[!exaple]- Non c’è spazio sufficiente
>Immaginiamo di voler inserire il record con chiave $25$
>
>![[Pasted image 20241214174642.png]]
>

Immaginiamo di voler inserire il record con chiave $40$ nel seguente b-tree

![[Pasted image 20241214165932.png|450]]

Ogni blocco del file principale deve contenere almeno $2$ record

![[Pasted image 20241214170106.png|500]]

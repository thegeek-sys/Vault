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
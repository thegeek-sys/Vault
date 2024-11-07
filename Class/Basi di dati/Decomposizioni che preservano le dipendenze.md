---
Created: 2024-11-07
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Cosa significa preservare le dipendenze?
Uno schema tipicamente viene decomposto per due motivi:
- **non è in 3NF**
- per motivi di efficienza degli accessi → infatti più è piccola la taglia delle tuple maggiore è il numero che riusciamo a caricare in memoria nella stessa operazione di lettura; se le informazioni della tupla non vengono utilizzate dallo stesso tipo di operazioni nella base di dati meglio decomporre lo schema

Abbiamo visto che quando uno schema viene decomposto, non basta che i sottoschemi siano in 3NF

---
## Decomposizione di uno schema di relazione

>[!info] Definizione
>Sia $R$ uno schema di relazione. Una **decomposizione** di $R$ è una famiglia $\rho \{R_{1},R_{2},\dots,R_{k}\}$ di sottoinsiemi di $R$ che ricopre $R$, ovvero che $\cup_{i=1}^k R_{i}=R$ (i sottoinsiemi possono avere intersezione non vuota)

In altre parole: se lo schema $R$ è composto da un certo insieme di attributi, decomporlo significa definire dei sottoschemi che contengono ognuno un sottoinsieme degli attributi di $R$.
I sottoschemi possono avere attributi in comune, e la loro unione deve necessariamente contenere tutti gli attributi di $R$

Quindi $R$ è un insieme di attributi, una decomposizione di $R$ è una famiglia di insiemi di attributi

>[!warning]
>Decomporre una istanza di una relazione con un certo schema, in base alla decomposizione dello schema stesso, significa proiettare ogni tupla dell’istanza originaria sugli attributi dei singoli sottoschemi eliminando i duplicati che potrebbero essere generati dal fatto che due tuple sono distinte ma hanno una posizione comune che ricade nello stesso schema
>
>>[!example]
>>![[Pasted image 20241107215231.png]]

---
## Equivalenza tra due insiemi di dipendenze funzionali

>[!info] Definizione
>Siano $F$ e $G$ due insiemi di dipendenze funzionali. $F$ e $G$ sono **equivalenti** ($F\equiv G$) se $F^+=G^+$
>
>>[!warning]
>>$F$ e $G$ non contengono le stesse dipendenze, ma le loro chiusure si


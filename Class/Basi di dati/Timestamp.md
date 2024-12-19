---
Created: 2024-12-20
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Il **timestamp** identifica una transazione e gli è assegnato dallo scheduler quando la transazione ha inizio. Questo può essere:
- il valore di un contatore (più è alto il valore più la transazione è “giovane“)
- l’ora di inizio della transazione

>[!info] Osservazione
>Se il timestamp della transazione $T_{1}$ è minore del timestamp della transazione $T_{2}$, la transazione $T_{1}$ è  iniziata prima della transazione $T_{2}$.
>
>Quindi se la transizioni non fossero eseguire in modo concorrente ma seriale, $T_{1}$ verrebbe eseguita prima di $T_{2}$

---
## Serializzabilità
Uno schedule è serializzabile se è equivalente allo schedule seriale in cui le transazioni compaiono ordinate in base al loro timestamp

Quindi uno schedule è serializzabile se per ciascun item acceduto da più di una transazione, l’ordine con cui le transazioni accedono all’item è quelli imposto dal timestamp

### Esempi

> [!example] Esempio 1
> Date $T_{1}$, $T_{2}$ e i loro timestamp $TS(T_{1})=110$ e $TS(T_{2})=100$
> ![[Pasted image 20241220004123.png|center|250]]
> 
> Quindi lo schedule è serializzabile se è equivalente allo schedule seriale $T_{2}T_{1}$
> ![[Pasted image 20241220004232.png|200]]
> 
> Consideriamo il seguente schema
> ![[Pasted image 20241220004314.png|200]]
> Lo schedule non è serializzabile in quanto $T_{1}$ legge $X$ prima che $T_{2}$ l’abbia scritto

> [!example] Esempio 2
> Date $T_{1}$, $T_{2}$ e i loro timestamp $TS(T_{1})=110$ e $TS(T_{2})=100$
> ![[Pasted image 20241220004538.png|center|250]]
> 
> Quindi lo schedule è serializzabile se è equivalente allo schedule seriale $T_{2}T_{1}$
> ![[Pasted image 20241220004619.png|200]]
> 
> Consideriamo il seguente schema
> ![[Pasted image 20241220004637.png|200]]
> Lo schedule non è serializzabile in quanto il valore di $X$ viene scritto da $T_{2}$ al posto che da $T_{1}$

---
## read timestamp, write timestamp
A ciascun item $X$ vengono associati due timestamp:
- **read timestamp** di $X$ ($read\_TS(X)$) → il valore più grande fra tutti i timestamp delle transazioni che hanno letto con successo $X$
- **write timestamp** di $X$ ($read\_WS(X)$) → il valore più grande fra tutti i timestamp delle transazioni che hanno scritto con successo $X$

### Esempi
Tornando agli esempi di prima

>[!example] Esempio 1
>![[Pasted image 20241220005213.png|center|500]]

>[!example] Esempio 2
>![[Pasted image 20241220005112.png|center|500]]

---
## Controllo della serializzabilità
Ogni volta che una transazione $T$ cerca di eseguire un $read(X)$ o un $write(X)$, occorre confrontare il timestamp $TS(T)$ di $T$ con il read timestamp e il write timestamp di $X$ per assicurarsi che l’ordine basato sui timestamp non è violato

### Algoritmo
$T$ cerca di eseguire una $write(X)$:
1. $read\_WS(X)>TS(T)$ → $T$ viene rolled back
2. $write\_WS(X)>TS(T)$ → l’operazione di scrittura non viene effettuata
3. se nessuna delle condizioni precedenti è soddisfatta allora
	- $write(X)$ è eseguita
	- $write\_TS(X):=TS(T)$

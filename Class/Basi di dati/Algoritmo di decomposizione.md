---
Created: 2024-11-22
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Qui mostreremo che dato uno schema di relazione $R$ e un insieme di dipendenze funzionali $F$ su $R$ esiste sempre una decomposizione $\rho=\{R_{1},R_{2},\dots,R_{k}\}$ di $R$ tale che:
- per ogni $i$, $i=1,\dots,k$, $R_{i}$ è in 3NF
- $\rho$ preserva $F$
- $\rho$ ha un join senza perdita
- tale decomposizione può essere calcolata in tempo polinomiale
### Come si fa?
Il seguente algoritmo, dato uno schema di relazione $R$ e un insieme di dipendenze funzionali $F$ su $R$, che è una copertura minimale, permette di calcolare in tempo polinomiale una decomposizione $\rho\{R_{1},R_{2},\dots,R_{k}\}$ di $R$ che rispetta le condizioni sopraelencate

Ci interessa una qualunque copertura minimale dell’insieme di dipendenze funzionali definite sullo schema $R$. Se ce ne fosse più di una, con eventualmente cardinalità diversa, potremmo scegliere ad esempio quella con meno dipendenza, ma questo non è tra i nostri scopi.
Quindi per fornire l’input all’algoritmo di decomposizione è sufficiente trovarne una tra quelle possibili. Poi vedremo perché ci occorre che sia una copertura minimale

---
## Algoritmo per la decomposizione di uno schema
$$
\begin{align}
\mathbf{Input}\quad&\text{uno schema di relazione }R\text{ e un insieme }F\text{ di dipendenze funzionali su } R\text{,}\\&\text{che è una copertura minimale} \\
\mathbf{Output}\quad&\text{una decomposizione }\rho \text{ di }R\text{ che preserva } F\text{ e tale che per ogni schema di}\\&\text{relazione in }\rho \text{ è in 3NF}
\end{align}
$$
$$
\begin{align}
&\mathbf{begin} \\
&S:=\varnothing \\
&\mathbf{for\,\,every} A\in R\text{ tale che }A\text{ non è coinvolto in nessuna dipendenza funzionale in F} \\
&\qquad\mathbf{do} \\
&\qquad S:=S\cup \{A\} \\
&\mathbf{if\,\,}S\neq \varnothing\mathbf{\,\,then} \\
&\qquad \mathbf{begin} \\
&\qquad R:=R-S \\
&\qquad \rho:=\rho \cup \{S\} \\
&\qquad \mathbf{end} \\
&\mathbf{if}\text{ esiste una dipendenza funzionale in }F\text{ che coinvolge tutti gli attributi in }R \\
&\qquad\mathbf{then\,\,}\rho:=\rho \cup \{R\} \\
&\mathbf{else} \\
&\qquad\mathbf{for\,\,every\,\,}X\to A \\
&\qquad\qquad\mathbf{do} \\
&\qquad\qquad \rho:=\rho \cup \{XA\} \\
&\mathbf{end}
\end{align}
$$
>[!hint]
$\mathbf{if}\text{ esiste una dipendenza funzionale in }F\text{ che coinvolge tutti gli attributi in }R$
>- $R$ residuo dopo aver eventualmente eliminato gli attributi inseriti prima in $S$
>
>$\mathbf{then\,\,}\rho:=\rho \cup \{R\}$
>- in questo caso ci fermiamo anche se la copertura minimale contiene anche altre dipendenze; in altre parole la copertura minimale potrebbe contenere anche altre dipendenze

---
## Teorema
Sia $R$ uno schema di relazione ed $F$ un insieme di dipendenze funzionali su $R$, che è una copertura minimale. L’algoritmo di decomposizione permette di calcolare in tempo polinomiale una decomposizione $\rho$ di $R$ tale che:
- ogni schema di relazione $\rho$
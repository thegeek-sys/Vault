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
- ogni schema di relazione $\rho$ è in 3NF
- $\rho$ preserva $F$

>[!info] Dimostrazione
>Dimostriamo separatamente le due proprietà della decomposizione
>
>##### $\rho$ preserva $F$
>Sia $G=\cup_{i=1}^k \pi_{R_{i}}(F)$, ovvero l’insieme delle dipendenze di $F^+$ tali che il determinante e il determinato appartengono al sottoschema.
>Poiché per ogni dipendenza funzionale $X\to A\in F$ si ha che $XA\in \rho$ (è proprio uno dei sottoschemi), si ah che questa dipendenza di $F$ sarà sicuramente in $G$, quindi $F\subseteq G$ e, quindi $F^+\subseteq G^+$. L’inclusione $G^+\subseteq F^+$ è banalmente verificata in quanto per definizione, $G\subseteq F^+$
>
>##### Ogni schema di relazione in $\rho$ è in 3NF
>Analizziamo i diversi casi che si possono presentare
>1. Se $S \in \rho$, ogni attributo in $S$ (elementi non coinvolti nelle dipendenze, e siccome la chiave deve determinare tutto lo schema, dovranno necessariamente essere nella chiave che li determinerà per riflessività) fa parte della chiave e quindi, banalmente, $S$ è in 3NF
>2. Se $R\in \rho$ esiste una dipendenza funzionale in $F$ che coinvolge tutti gli attributi in $R$. Poiché $F$ è una copertura minimale tale dipendenza avrà la forma $R-A\to A$. Ma se fosse esistito $Y\to A$ con $Y\subset R-A$ allora nella copertura non ci sarebbe stato $R-A\to A$
>3. Se $XA\in \rho$, poiché $F$ è una copertura minimale, non ci possono essere una dipendenza funzionale $X'\to A\in F^+$ tale che $X'\subset X$ e, quindi, $X$ è chiave in $XA$. Sia $Y\to B$ una qualsiasi dipendenza in $F$ tale che $YB\subseteq XA$; se $B=A$ allora, poiché $F$ è una copertura minimale, $Y=X$ (cioè, $Y$ è superchiave); se $B\neq A$ allora $B\in X$ e quindi $B$ è primo

---
## Teorema
Sia $R$ uno schema di relazione, $F$ un insieme di dipendenze funzionali su $R$, che è una copertura minimale e $\rho$ la decomposizione di $R$ prodotta dall’algoritmo di decomposizione. La decomposizione $\sigma=\rho \cup \{K\}$, dove $K$ è una chiave per $R$, è tale che:
- ogni schema di relazione in $\sigma$ è in 3NF
- $\sigma$ preserva $F$
- $\sigma$ ha un join senza perdita

>[!info] Dimostrazione
>##### $\sigma$ preserva $F$
>Poiché $\rho$ preserva $F$ anche $\sigma$ preserva $F$
>Stiamo aggiungendo un nuovo sottoschema, quindi alla nuova $G'$ dobbiamo aggiungere una proiezione di $F$, cioè $G'=G\cup \pi_{K}(F)$ quindi $G'\supseteq G\supseteq F$ e quindi $G'^+\supseteq G^+\supseteq F^+$
>L’inclusione $G'^+\subseteq F^+$ è di nuovo banalmente verificata in quanto, per definizione, $G\subseteq F^+$
>
>##### Ogni schema di relazione in $\sigma$ è in 3NF
>Poiché $\sigma=\rho \cup \{K\}$, è sufficiente verificare che anche lo schema di relazione $K$ è in 3NF. Mostriamo che $K$ è chiave anche per lo schema $K$.
>Supponiamo per assurdo che $K$ non sia chiave per lo schema $K$; allora esiste un sottoinsieme proprio $K'$ di $K$ che determina tutto lo schema $K$, cioè tale che $Kì\to K\in F^+$ (più precisamente alla chiusura di $\pi_{K}(F)$, ma poiché $\pi_{K}(F)\subset F^+$ allora $(\pi_{K}(F))^+\subset F^+$).
>Poiché $K$ è chiave per lo schema $R$, $K\to R\in F^+$, pertanto per transitività $K'\to R\in F^+$, che contraddice il fatto che $K$ è chiave per lo schema $R$ (verrebbe violato il requisito di minimalità).
>Pertanto $K$ è chiave per lo schema $K$ e quindi per ogni dipendenza funzionale $X\to A\in F^+$ con $XA\subseteq K$, $A$ è primo


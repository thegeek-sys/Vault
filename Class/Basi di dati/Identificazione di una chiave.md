---
Created: 2024-11-07
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Chiavi di uno schema di relazione
Utilizziamo il calcolo della chiusura di un insieme di attributi per determinare le chiavi di uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$

>[!example]
>$$R=(A,B,C,D,E,H)$$
>$$F=\{AB\to CD,C\to E,AB\to E, ABC\to D\}$$
>
>Calcolare la chiusura dell’insieme $ABH$
>
>$$
\begin{flalign}
&\text{begin}\\
&Z:=\color{red}ABH\\
&S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}=\{C \text{ (per la dipendenza }\textcolor{red}{AB}\to\textcolor{royalblue}{CD}\text{)},\\& E\text{ (per la dipendenza }\textcolor{red}{AB}\to\textcolor{royalblue}{E}\text{)}\}=\textcolor{royalblue}{CDE}\\
&\text{while } S\not\subset Z\,\,\,\text{(}CDE \not\subset ABH\text{ quindi entriamo nella prima iterazione)}\\
&\qquad\text{do}\\
&\qquad\text{begin}\\
&\qquad\qquad Z:=Z\cup S=ABH\cup CDE=ABCDEH\\
&\qquad\qquad S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}=\{C \text{ (per la }\\&\qquad\qquad\text{dipendenza }\textcolor{red}{AB}\to \textcolor{royalblue}{E}\text{)}, D \text{ (per la dipendenza }\textcolor{red}{AB}\to \textcolor{royalblue}{CD}\text{)},\\&\qquad\qquad E \text{ (per la dipendenza }\textcolor{red}{AB}\to \textcolor{royalblue}{E}\text{)}\}\\
&\qquad\text{end}\\
&\qquad\text{controllo del while:}CDE\subset ABCDEH \text{(non abbiamo aggiunto nulla di nuovo)}\\
&\qquad\text{usciamo dal while}\\
&\text{end}\\
\end{flalign}\\
>$$
>$$ABH^+=ABCDEH=R$$
>
>Verifichiamo se $ABH$ può essere chiave
>
>**Ricordiamo le condizioni**
>Un insieme $F$ di attributi per essere chiave di una relazione $R$ se:
>- $K\to R \in F^+$
>- non esiste un sottoinsieme proprio $K'$ di $K$ tale che $K'\to R\in F^+$
>
>Utilizzo le osservazioni sotto
>$H$ non compare nelle dipendenze quindi necessariamente si trova nella chiave
>Inoltre $A$ e $B$ non sono mai determinati da nessuna dipendenza, dunque anch’essi si devono trovare nella chiave
>
>>[!hint]
>>All’esame evitare di saltare i passaggi ma evitare anche di fare i calcoli espliciti. Per verificare se un insieme di attributi è una chiave o una superchiave spiegare i ragionamenti fatti attraverso le osservazioni

---
## Osservazioni
### Osservazione 1
Per verificare se un insieme di attributi è chiave conviene partire dai sottoinsiemi con cardinalità maggiore, se la loro chiusura non contiene $R$, è inutile calcolare la chiusura dei loro rispettivi sottoinsiemi
### Osservazione 2
Gli attributi che non compaiono mai a destra delle dipendenze funzionali di $F$, non sono determinati funzionalmente da nessun altro attributo quindi rimarrebbero fuori dalla chiusura  di qualunque sottoinsieme di $R$ che non lo contenesse ma ogni chiave deve determinare tutto $R$, quindi gli attributi che **non compaiono a destra di nessuna dipendenza funzionale in $F$ dovranno essere sicuramente in ogni chiave**
### Osservazione 3
Gli attributi che non compaiono mai nelle dipendenze funzionali di $F$, non sono determinati funzionalmente da nessun altro attributo quindi rimarrebbero fuori dalla chiusura  di qualunque sottoinsieme di $R$ che non lo contenesse ma ogni chiave deve determinare tutto $R$, quindi gli attributi che **non compaiono in nessuna dipendenza funzionale in $F$ dovranno essere sicuramente in ogni chiave**
### Osservazione 3
Le osservazioni 1 e 2 valgono anche quando stiamo cercando la/le chiave/i di uno schema
### Osservazione 4
L’approccio di forza bruta (provate tutti i sottoinsiemi) non è sbagliato ma molto poco efficiente

---
## Esempi

>[!example]
>$$R=(A,B,C,D,E,G,H)$$
>$$F=\{AB\to D, G\to A, G\to B, H\to E,H\to G,D\to H\}$$
>
>Le chiavi sono:
>- $K_{1}=(GC)$
>- $K_{2}=(ABC)$
>- $K_{3}=(DC)$
>- $K_{4}=(CH)$

>[!example]
>$$R=(A,B,C,D,E)$$
>$$F=\{AB\to C, AC\to B, D\to E\}$$
>
>Le chiavi sono:
>- $K_{1}=(ABD)$
>- $K_{2}=(ACD)$

---
## Test di unicità di una chiave
Dati uno schema di relazione $R$ e un insieme di dipendenze funzionali $F$, calcoliamo l’intersezione degli insiemi ottenuti come sopra, cioè degli insiemi $X=R-(W-V)$ con $V\to W \in F$
Se l’intersezione di questi insiemi determina tutto $R$, allora questa intersezione è l’unica chiave di $R$

>[!example] Esempio precedente
>- $ABCDE-(C-AB)=ABDE$
>- $ABCDE-(B-AC)=ACDE$
>- $ABCDE-(E-D)=ABCD$
>
>$$(ABDE\cap ACDE\cap ABCD)^+=(AD)^+=AD$$
>
>Quindi avremmo già potuto capire che esiste più di una chiave
>- Se l’intersezione di questi insiemi non determina tutto $R$ allora esistono più chiavi che vanno tutte identificate per il test 3NF
>
>>[!hint]
>>Questa è una comoda verifica, ma se nel compito viene richiesto di **verificare che un insieme di attributi sia chiave, o di trovare la chiave, va usata solo la definizione** (verifica che la chiusura contenga $R$ e che nessun sottoinsieme abbia la stessa proprietà)

---
## Chiavi  e 3NF
Una volta individuate le chiavi di uno schema di relazione, possiamo determinare se lo schema è in 3NF

>[!example]
>$$R=(A,B,C,D,E,G,H)$$
>$$F=\{AB\to D, G\to A, G\to B, H\to E,H\to G,D\to H\}$$
>
>Le chiavi sono:
>- $K_{1}=(GC)$
>- $K_{2}=(ABC)$
>- $K_{3}=(DC)$
>- $K_{4}=(CH)$
>
>Non è in 3NF in quanto $H$ non è superchiave e $E$ non è primo


>[!example]
>$$R=(A,B,C,D,E,G,H)$$
>$$F=\{AB\to CD, EH\to D, D\to H\}$$
>
>Chiavi:
>- $K=(ABEG)$
>
>Non è in 3NF in quanto $AB$ non è superchiave e $CD$ non è primo
>
>>[!warning]
>>Anche se si trova che uno schema appare in 3NF devo verificare che anche le dipendenze in $F^+$ rispettano la 3NF


>[!example]
>$$R=(A,B,C,D,E,H)$$
>$$F=\{AB\to CD, C\to E, AB\to E, ABC\to D\}$$
>
>$K=(ABH)$
>Non è in 3NF dato che $AB$ non è superchiave e $CD$ non è primo

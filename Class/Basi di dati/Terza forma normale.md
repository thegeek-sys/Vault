---
Created: 2024-10-27
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Quali sono i problemi di uno schema mal progettato?
Ritorniamo al nostro esempio di base di dati che contiene le informazioni sugli studenti e sugli esami sostenuti, e ripartiamo dalla soluzione “buona” tornata alla fine

![[Progettazione di una base di dati relazionale - Problemi e vincoli#Ipotesi 3]]

### Considerazioni
Poiché il numero di matricola identifica univocamente uno studente, ad ogni numero di matricola corrisponde:
- un solo codice fiscale ($\text{Matr}\to \text{CF}$)
- un solo cognome ($\text{Matr}\to \text{Cogn}$)
- un solo nome ($\text{Matr}\to \text{Nome}$)
- una sola data di nascita ($\text{Matr}\to \text{Data}$)
- un solo comune di nascita ($\text{Matr} \to \text{Com}$)

Quindi un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{Matr}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$
Con considerazioni analoghe abbiamo che un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{CF}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$
Pertanto sia $\text{Matr}$ che $\text{CF}$ sono chiavi di $\text{Studente}$

D’altra parte possiamo osservare che ci possono essere due studenti con lo stesso cognome e nomi differenti quindi possiamo avere due istanze di $\text{Studente}$ che **non** soddisfano le dipendenza funzionale $\text{Cogn}\to \text{Nome}$. Inoltre possiamo avere istanze di $\text{Studente}$ che non soddisfano: $\text{Cogn}\to \text{Nome}$, $\text{Cogn}\to \text{Data}$, $\text{Cogn}\to \text{Com}$ ecc.

Con ciò possiamo concludere che le **uniche dipendenze funzionali non banali** che devono essere soddisfatte da un’istanza legale di $\text{Studente}$ sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave ($\text{Matr}$ o $\text{CF}$)

>[!hint]
>Vedremo che questa è la prima condizione che però va ulteriormente rifinita per arrivare ad una definizione precisa di terza forma normale (3NF)


$$
\text{Esame(Matr, C\#, Data, Voto)}
$$
Uno studente può sostenere l’esame relativo ad un corso una sola volta; pertanto per ogni esame esiste:
- una sola data (in cui è stato sostenuto)
- un solo voto
Quindi ogni istanza legale di $\text{Esame}$ deve soddisfate la dipendenza funzionale
$$
\text{Matr, C\#}\to \text{Data, Voto}
$$
D’altra parte uno studente può sostenere esami in dati differenti e riportare voti diversi nei vari esami. Pertanto esistono istanze di $\text{Esame}$ che non soddisfano una o entrambe le dipendenze funzionali: $\text{Matr}\to \text{Data}$, $\text{Matr}\to \text{Voto}$

Inoltre l’esame relativo ad un certo corso non può essere superato da diversi studenti in date diverse e con voti diversi. Pertanto esistono istanze di $\text{Esame}$ che non soddisfano una o entrambe le dipendenze funzionali: $\text{C\#}\to \text{Data}$, $\text{C\#}\to \text{Voto}$
Pertanto $\text{Matr, C\#}$ è una chiave per $\text{Esame}$

>[!hint]
>Vedremo in seguito delle procedure rigorose per identificare la/le chiavi

### Conclusioni
Per ciascun schema di relazione:
- $\text{Studente (Matr, CF, Cogn, Nome, Data, Com)}$
- $\text{Corso (C\#, Tit, Doc)}$
- $\text{Esame (Matr, C\#, Data, Voto)}$
- $\text{Comune (Com, Prov)}$

>[!info] Stiamo continuando ad assumere che $\text{Com}\to \text{Prov}$, cioè che non ci sono comuni omonimi

Le **uniche dipendenze funzionali non banali** che devono essere soddisfatte da ogni istanza legale di sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave

---
## Terza forma normale
Uno schema di relazione è in **3NF** se le uniche dipendenze funzionali non banali (non riflessive) che devono essere soddisfatte da ogni istanza legale sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave oppure $X$ è contenuto in una chiave

### Definizione
Dati uno shema di relazione $R$ e un insieme di dipendenze funzionali $F$ su $R$, $R$ è in **3NF** se
$$
\forall X\to A\in F^+ \,\,\,\,\,\,\,A\not\in X
$$
- $A$ appartiene ad una chiave (è **primo**, applicabile solo ai cingleton)
- $X$ contiene una chiave (è una superchiave)

>[!warning]
>- è sbagliato scrivere $\forall X\to A\in F$, perché non sapremmo se e come valutare una dipendenza del tipo $X\to AB$ (due o più attributi a destra)
>- se sostituisco $\forall X\to A\in F$ con $\forall X\to Y\in F$, non so come comportarmi se $Y$ contiene sia attributi primi che non
>- la condizione $A \not\in X$ è importante. Infatti, per l’assioma della riflessività, se $A\in X$ avremo sempre $X\to A \in F^A$ e quindi in $F^+$, anche quando $A$ non è primo e $X$ non è superchiave, e quindi se considerassimo questo tipo di dipendenze nessuno schema risulterebbe in 3NF

### Esempi
>[!example] Esempio 1
>$$
\begin{flalign}R=A,B,C,D&& F=\{A\to B, B\to BC\}\end{flalign}
>$$
>
>Ho come chiavi:
>- $K_{1}=A$ → $A\to CD$ per transitività e $A\to BCD$ per unione
>
>$A\to B$ → rispetta le condizioni per essere in 3NF
>$B\to BC$ → $B$ non è superchiave, controllo quindi i determinati. $C$ e $D$ violano la 3NF perché in entrambi i casi non fanno parte di una chiave
>
>Lo schema non è in 3NF

>[!example] Esempio 2
>$$
\begin{flalign}R=A,B,C,D&& F=\{AB\to CD, AC\to BD, D\to BC\}\end{flalign}
>$$
>
>Ho come chiavi:
>- $K_{1}=AB$
>- $K_{2}=AC$
>- $K_{3}=AD$ (per il teorema dell’aumento sull’ultima dipendenza funzionale)
>
>
>D nell’ultima dipendenza non è una chiave (ma un pezzo di una chiave) però il determinato è composto da $B$ (un attributo della chiave $AB$) e da $C$ (un attributo della chiave $AC$); dunque lo schema è in **3FN**.
>
>Ho infatti decomposto in $D\to B$ ($B$ è parte di una chiave) e in $D\to C$ ($C$ è una parte di una chiave)


---
## Dipendenza parizale
Siano $R$ uno schema di relazione e $F$ un insieme di dipendenze funzionali su $R$.
$X\to A\in F^+\mid A\not\in X$ è una **dipendenza parziale** su $R$ se $A$ non è primo ed $X$ è contenuto propriamente in una chiave di $R$
![[Pasted image 20241027185741.png|250]]

## Dipendenza transitiva
Siano $R$ uno schema di relazione e $F$ un insieme di dipendenze funzionali su $R$.
$X\to A\in F^+\mid A\not\in X$ è una **dipendenza transitiva** su $R$ se $A$ non è primo e per ogni chiave $K$ di $R$ si ha che $X$ non p contenuto propriamente in $K$ e $K-X\neq\varnothing$

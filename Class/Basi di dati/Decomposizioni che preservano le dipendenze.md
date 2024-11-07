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

---
## Che si fa?
Verificare l’equivalenza di due insiemi $F$ e $G$ di dipendenze funzionali richiede che venga verificata l’uguaglianza di $F^+$ e $G^+$, cioè che $F^+\subseteq G^+$ e che $F^+\supseteq G^+$

Come detto in precedenza, calcolare la chiusura di un insieme di dipendenze funzionali richiede tempo esponenziale. Il seguente lemma ci permette tuttavia di verificare l’equivalenza dei due insiemi di dipendenze funzionali in tempo polinomiale

---
## Lemma 2
Siano $F$ e $G$ due insiemi di dipendenze funzionali. Se $F\subseteq G^+$ allora $F^+\subseteq G^+$

>[!info] Dimostrazione
>Sia $f\in F^+ - F$ (è una dipendenza in $F^+$ che non compare in $F$)
>
>Ogni dipendenza in $F$ è derivabile da $G$ mediante gli assiomi di Armstrong (per ipotesi $F$ si trova in $G^+$)
>Inoltre $f\in F^+$ è derivabile dalle dipendenze in $F$ mediante gli assiomi di Armstrong
>Dunque si può dire che $f$ è derivabile da $G$ mediante gli assiomi di Armstrong
>$$G\overset{A}{\longrightarrow} F\overset{A}{\longrightarrow} F^+$$

---
## Preservare le dipendenze funzionali

>[!info] Definizione
>Sia $R$ uno schema di relazione, $F$ un insieme di dipendenze funzionali su $R$ e $\rho \{R_{1},R_{2},\dots R_{k}\}$ una decomposizione di $R$.
>Diciamo che $\rho$ **preserva** $F$ se $F\equiv \cup_{i=1}^k \pi_{R_{i}}(F)$ dove $\pi_{R_{i}}(F)=\{X\to Y \text{ t.c. }X\to Y \in F^+\land XY\subseteq R_{i}\}$

>[!warning]
>Ovviamente $\cup_{i=1}^k \pi_{R_{i}}(F)$ è un insieme di dipendenze funzionali 
>Ogni $\pi_{R_{i}}(F)$ è un insieme di dipendenze funzionali dato dalla proiezione dell’insieme di dipendenze funzionali $F$ sul sottoschema $R_{i}$
>
>Proiettare un insieme di dipendenze $F$ su un sottoschema $R_{i}$ non significa banalmente perdere le dipendenze dell’insieme $F$ ed eliminare da queste dipendenze gli attributi che non sono in $R_{i}$, ma **prendere tutte e sole** le dipendenze **derivabili da $F$** tramite gli assiomi di Armstrong (quindi quelle in $F^+$) che hanno tutti gli attributi (dipendenti e determinanti) in $R_{i}$

---
## Verificare validità di una decomposizione
Supponiamo di avere già una decomposizione e di voler verificare se preserva le dipendenze funzionali
Per fare ciò deve essere verificata l’equivalenza dei due insiemi di dipendenze funzionali $F$ e $G=\cup_{i=1}^k \pi_{R_{i}}(F)$ e quindi la doppia inclusione $F^+\subseteq G^+$ e $F^+\supseteq G^+$

Per come è definito $G$ in questo caso sarà sicuramente $F^+\supseteq G^+$, infatti ogni proiezione di $F$ che viene inclusa per definizione in $G$ è un sottoinsieme di $F^+$, quindi $F^+$ contiene $G$ e per il lemma 2 questo implica che $G^+\subseteq F^+$

Dunque non dobbiamo verificare una delle due implicazioni, inoltre per il lemma 2 $F\subseteq G^+$ implica che $F^+\subseteq G^+$ dunque ci basta verificare che: $F\subseteq G^+$ semplificandoci di molto il lavoro

Questa verifica può essere fatta con l’algoritmo che segue
$$
\begin{align}
&\mathbf{Input}\to \text{due insiemi }F \text{ e }G\text{ di dipendenze funzionali su R} \\
&\mathbf{Output}\to \text{la variabile successo (true se }F\subseteq G^+\text{)}\\
&\text{begin}\\
&\qquad\text{successo}:=true\\
&\qquad \text{for every }X\to Y\in F\\
&\qquad\text{do}\\
&\qquad \text{begin}\\
&\qquad\qquad\text{calcola } X^+\\
&\qquad\qquad \text{if }Y\not\subset X^+_{G} \text{ then successo}:=false\\
&\qquad\text{end} \\
&\text{end}
\end{align}
$$

Il problema di questo algoritmo è che dovremmo prima calcolare $G$, ma per definizione di $G$ ciò richiede il calcolo di $F^+$ che richiede tempo esponenziale.

Presentiamo un algoritmo che permette di calcolare $X^+_{G}$ a partire da $F$

$$
\begin{align}
\mathbf{Input}\to \text{ }& \text{uno schema } R\text{, un insieme }F\text{ di dipendenze funzionali su }R, \\
&\text{una decomposizione }\rho=\{R_{1},R_{2},\dots,R_{k}\}\text{ di }R,\text{un sottoinsieme }X\text{ di }R \\\\
\mathbf{Output}\to \text{ }&\text{la chiusura di }X\text{ rispetto a } G=\cup_{i=1}^k\pi_{R_{i}}(F)\text{ (nella variabile }Z\text{)}
\end{align}
$$
$$
\begin{align}
&\text{begin}\\
&\qquad Z:=X \\
&\qquad S:=\varnothing \\
&\qquad \text{for }i:=1\text{ to }k\\
&\qquad\text{do}\qquad S:=S\cup(Z\cap R_{i})^+_{F}\cap R_{i}  \\
&\qquad\text{while } S\not\subset Z \\
&\qquad\qquad\text{do}\\
&\qquad\qquad \text{begin}\\
&\qquad\qquad\qquad Z:=Z\cup S\\
&\qquad\qquad\qquad \text{for }i:=1\text{ to }k \\
&\qquad\qquad\qquad\text{do}\qquad S:=S\cup(Z\cap R_{i})^+_{F}\cap R_{i} \\
&\qquad\qquad\text{end} \\
&\text{end}
\end{align}
$$


>[!info]
>Con $S:=S\cup(Z\cap R_{i})^+_{F}\cap R_{i}$ sostanzialmente si calcola la chiusura in $F$ degli elementi (di cui cerchiamo di calcola la chiusura in $G$) rispetto al sottoschema $R_{i}$, infine facciamo l’intersezione con $R_{i}$ in modo tale da avere al massimo tutti gli attributi contenuti di $R_{i}$.
>In questo modo rispettiamo la definizione di $G=\cup_{i=1}^k\pi_{R_{i}}(F)$


ma visto che G equiv ad F dove stanno le dipendenze “a cavallo”?
in G sono contenute tutte e solo le dipendenze dei singoli sottoschemi (quindi non le dipendenze che hanno un attributo in un sottoschema e un attributo in un altro), se questo è vero vuol dire che non sono preservate le dipendenze in F (vedi definizione)


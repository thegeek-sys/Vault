---
Created: 2024-10-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Introduciamo $\text{F}^\text{A}$|Introduciamo $\text{F}^\text{A}$]]
- [[#Assiomi di Armstrong|Assiomi di Armstrong]]
	- [[#Assiomi di Armstrong#Assioma della riflessività|Assioma della riflessività]]
		- [[#Assioma della riflessività#Esempio|Esempio]]
	- [[#Assiomi di Armstrong#Assioma dell’aumento|Assioma dell’aumento]]
		- [[#Assioma dell’aumento#Esempio|Esempio]]
	- [[#Assiomi di Armstrong#Assioma della transitività|Assioma della transitività]]
		- [[#Assioma della transitività#Esempio|Esempio]]
- [[#Conseguenze degli assiomi di Armstrong|Conseguenze degli assiomi di Armstrong]]
	- [[#Conseguenze degli assiomi di Armstrong#Regola dell’unione|Regola dell’unione]]
	- [[#Conseguenze degli assiomi di Armstrong#Regola della decomposizione|Regola della decomposizione]]
	- [[#Conseguenze degli assiomi di Armstrong#Regola della pseudotransitività|Regola della pseudotransitività]]
	- [[#Conseguenze degli assiomi di Armstrong#Osservazione|Osservazione]]
- [[#Chiusura di un insieme di attributi|Chiusura di un insieme di attributi]]
	- [[#Chiusura di un insieme di attributi#Determinare la chiave di una relazione|Determinare la chiave di una relazione]]
- [[#Lemma 1|Lemma 1]]
- [[#Teorema: $F^+=F^A$|Teorema: $F^+=F^A$]]
	- [[#Teorema: $F^+=F^A$#Nota finale|Nota finale]]
- [[#A cosa serve conoscere $F^+$?|A cosa serve conoscere $F^+$?]]
---
## Introduciamo $\text{F}^\text{A}$
Ricordiamo che il nostro problema è calcolare l’insieme di dipendenze $F^+$ che viene **soddisfatto da ogni istanza legale** di uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$.
Abbiamo concluso che banalmente $F\subseteq F^+$ in quanto una istanza è legale solo se soddisfa tutte le dipendenze in $F$

---
## Assiomi di Armstrong
Denotiamo con $F^A$ l’insieme di dipendenze funzionali definito nel modo seguente:
- se $f \in F$ allora $f \in F^A$
- se rispetta l’**assioma della riflessività** (determina le dipendenze funzionali banali)
- se rispetta l’**assioma dell’aumento**
- se rispetta l’**assioma della transitività**

Dimostreremo che $\mathbf{F^+=F^A}$, cioè la chiusura di un insieme di dipendenze funzionali $F$ può essere ottenuta a partire da $F$ applicando ricorsivamente gli assiomi della riflessività, dell’aumento e della transitività, conosciuti come **assiomi di Armstrong**

### Assioma della riflessività
$$
\text{se } Y\subseteq X\subseteq R \text{ allora } X\rightarrow Y \in F^A
$$
#### Esempio
$\text{Nome}\subseteq(\text{Nome, Cognome})$ quindi ovviamente se due tuple hanno uguale la coppia $(\text{Nome, Cognome})$ allora sicuramente avranno uguale l’attributo $\text{Nome}$ (idem per $\text{Cognome}$), quindi $(\text{Nome, Cognome}) \rightarrow \text{Nome}$ viene sempre soddisfatta

### Assioma dell’aumento
$$
\text{se } X \to Y \in F^A \text{ allora } XZ \to YZ\in F^A, \text{ per ogni } Z \subseteq R
$$
#### Esempio
$\text{CodFiscale}\rightarrow\text{Cognome}$ è soddisfatta quando, se due tuple hanno $\text{CodFiscale}$ uguale, allora hanno anche $\text{Cognome}$ uguale.
Se la dipendenza è soddisfatta, e aggiungo l’attributo $\text{Indirizzo}$, avrò che se due tuple sono uguali su $(\text{CodFiscale, Indirizzo})$ lo devono essere anche su $(\text{Cognome, Indirizzo})$ ($\text{Indirizzo}$ è incluso nella porzione di tuple che è uguale), quindi se viene soddisfatta $\text{CodFiscale}\rightarrow\text{Cognome}$ viene soddisfatta anche $\text{CodFiscale, Indirizzo}\rightarrow\text{Cognome, Indirizzo}$

### Assioma della transitività
$$
\text{se } X\to Y\in F^A \text{ e } Y\to Z\in F^A\text{ allora }X\to Z\in F^A
$$
#### Esempio
$\text{Matricola}\rightarrow\text{CodFiscale}$ è soddisfatta quando, se due tuple hanno $\text{Matricola}$ uguale, allora hanno anche $\text{CodFiscale}$ uguale
$\text{CodFiscale}\rightarrow\text{Cognome}$ è soddisfatta quando, se due tuple hanno $\text{CodFiscale}$ uguale, allora hanno anche $\text{Cognome}$ uguale
Allora se entrambe le dipendenze sono soddisfatte, e due tuple hanno $\text{Matricola}$ uguale, allora hanno anche $\text{CodFiscale}$ uguale, ma allora hanno anche $\text{Cognome}$ uguale, quindi se entrambe le dipendenze sono soddisfatte, ogni volta che due tuple hanno $\text{Matricola}$ uguale avranno anche $\text{Cognome}$ uguale, e quindi viene soddisfatta anche $\text{Matricola}\rightarrow\text{Cognome}$

---
## Conseguenze degli assiomi di Armstrong
Prima di procedere introduciamo altre tre regole conseguenza degli assiomi che consentono di derivare da dipendenze funzionali in $F^A$ altre dipendenze funzionali in $F^A$

### Regola dell’unione
$$
\text{se } X\rightarrow Y \in F^A \text{ e } X\rightarrow Z \in F^A \text{ allora } X \rightarrow YZ \in F^A
$$

>[!info] Dimostrazione
>- Se $X\rightarrow Y \in F^A$, per l’assioma dell’aumento si ha $X\rightarrow XY \in F^A$
>- Analogamente se $X\rightarrow Z \in F^A$, per l’assioma dell’aumento si ha $XY \rightarrow YZ \in F^A$
>- Quindi poiché $X\rightarrow XY \in F^A$ e $XY \rightarrow YZ \in F^A$, per l’assioma della transitività si ha $X\rightarrow YZ \in F^A$
>$\begin{flalign}&& \blacksquare\end{flalign}$

### Regola della decomposizione
$$
\text{se }X \rightarrow Y \in F^A\text{ e }Z \subseteq Y\text{ allora }X\rightarrow Z \in F^A
$$

>[!info] Dimostrazione
>- Se $Z \subseteq Y$ allora per l’assioma della riflessività, si ha $Y\to Z\in F^A$
>- Quindi poiché $X\to Y\in F^A$ e $Y\to Z\in F^A$ per l’assioma della transitività si ha $X\to Z\in F^A$
>$\begin{flalign}&& \blacksquare\end{flalign}$

### Regola della pseudotransitività
$$
\text{se }X \rightarrow Y \in F^A\text{ e }WY \rightarrow Z \in F^A\text{ allora }WX \rightarrow Z \in F^A
$$
>[!info] Dimostrazione
>- Se $X\to Y\in F^A$, per l’assioma dell’aumento si ha $WX\to WY\in F^A$
>- Quindi poiché $WX\to WY\in F^A$ e $WY\to Z\in F^A$, per l’assioma della transitività si ha $WX\to Z\in F^A$
>$\begin{flalign}&& \blacksquare\end{flalign}$

### Osservazione
Osserviamo che:
- per la regola dell’**unione**, se $X\to A_{i}\in F^A$, $i=1,\, \dots,\,n$ allora $X\to A_{1},\,\dots,\,A_{i}\,\dots\,A_{n}\in F^A$
- per la regola della **decomposizione**, se $X\to A_{1},\,\dots ,\,A_{i},\,\dots,\,A_{n}\in F^A$ allora $X\to A_{i}\in F^A,\,\, i=1,\,\dots,\,n$
quindi:
$$
	X\to A_{1},\,\dots,\,A_{i},\,\dots,\,A_{n}\in F^A \Leftrightarrow X\to A_{i}\in F^A, \,\,\, i=1,\,\dots,\,n
$$
e possiamo limitarci in generale a considerare la dipendenze col membro destro singleton

---
## Chiusura di un insieme di attributi
Dato $X$ un attributo di uno schema di relazione $R$ e $F$ un insieme di dipendenze funzionali su $R$ e $X$ un sottoinsieme di $R$.
La **chiusura di $\textcolor{SkyBlue}{\mathbf{X}}$** rispetto a $F$, denotata con $X^+_{F}$ (o semplicemente $X^+$ se non sorgono ambiguità), è definito nel modo seguente:
$$
X^+_{F}=\{A\mid X\to A\in F^A\}
$$

>[!hint]
>La chiusura di $X$ non può essere mai vuota, deve contenere almeno sé stesso
>$$X \subseteq X^+_{F}$$

In pratica la chiusura di un insieme di attributi $X$ contiene tutti gli attributi determinati **direttamente** o **indirettamente** da $X$, ovvero tutti quelli che sono determinati funzionalmente da $X$ eventualmente applicando gli assiomi di Armstrong


>[!example] Esempio
>- $\text{CF}\rightarrow\text{COMUNE}$
>- $\text{COMUNE}\rightarrow\text{PROVINCIA}$
>
>Dunque $\text{CF}\rightarrow\text{COMUNE}$ è diretta mentre $\text{CF}\rightarrow\text{PROVINCIA}$ è indiretta
>$$
>\text{CF}^+_{F}=\{\text{COMUNE, PROVINCIA, CF}\}
>$$


### Determinare la chiave di una relazione
La chiusura di un insieme di attributi può essere utile anche per determinare le chiavi di una relazione

>[!example] Esempio
>- $\text{Auto(MODELLO, MARCA, CILINDRATA, COLORE)}$
>- $F=\{\text{MODELLO} \rightarrow \text{MARCA}, \text{MODELLO}\rightarrow\text{COLORE}\}$
>
>Dunque come chiusure si ha:
>- $(\text{MODELLO})^+_{F}=\{\text{MODELLO, MARCA, COLORE}\}$
>- $(\text{MARCA})^+_{F}=\{\text{MARCA}\}$
>- $(\text{CILINDRATA})^+_{F}=\{\text{CILINDRATA}\}$
>- $(\text{COLORE})^+_{F}=\{\text{COLORE}\}$
>
>$$\text{chiave}=\text{MODELLO, CILINDRATA}$$

---
## Lemma 1
Siano $R$ uno schema di relazione ed $F$ un insieme di dipendenze funzionali su $R$.
Si ha che: $X\rightarrow Y \in F^A \Leftrightarrow Y \subseteq X^+$

>[!info] Dimostrazione
>Sia $Y=A_{1}, A_{2}, \dots, A_{n}$
>
>**Parte se**
>Poiché $Y \subseteq X^+$, per ogni $i$, $i=1,\, \dots,\, n$ si ha che $X\rightarrow A_{i} \in F^A$. Pertanto per la regola dell’unione, $X\rightarrow Y \in F^A$
>
>**Parte solo se**
>Poiché $X\rightarrow Y \in F^A$, per la regola della decomposizione si ha che, per ogni $i$, $i=1, \dots, n$, $X \rightarrow A_{i} \in F^A$, cioè $A_{i} \in X^+$ per ogni $i, i=1,\, \dots,\, n$, e, quindi, $Y \subseteq X^+$
>$\begin{flalign}&& \blacksquare\end{flalign}$

---
## Teorema: $F^+=F^A$
Siano $R$ uno schema di relazione ed $F$ un insieme di dipendenze funzionali su $R$.
Si ha $F^+=F^A$

>[!info] Dimostrazione
>Per dimostrare l’uguaglianza di due insiemi ci basta dimostrare la doppia inclusione
>$$F^A \subseteq F^+\land F^+ \subseteq F^A$$
>
>##### Dimostriamo che  $F^A\subseteq F^+$
>Sia $X\to Y$ una dipendenza funzionale in $F^A$. Dimostriamo che $X\to Y\in F^+$ per induzione sul numero $i$ di applicazioni di uno degli assiomi di Armstrong
>- Base dell’induzione ($i=0$): $X\to Y\in F\implies X\to Y\in F^+\,\,\,\, F\subseteq F^+$
>- Ipotesi induttiva ($i>0$): $X\to Y\in F^A\implies X\to Y\in F^+\implies X\to Y$ soddisfatto da ogni istanza legale
>- Passo $i$: $X\to Y\in F^A$ ottenuto in $i$ passi
><br>
>
>Si possono presentare tre casi
>1. $X\to Y$ ottenuto per **riflessività** $\implies Y\subseteq X$
>	$$\forall r \text{ (legale) } t_{1}[X]=t_{2}[X]\,\,\,Y\subseteq X \implies t_{1}[Y]=t_{2}[Y]\implies X\to Y\in F^+$$
>
>2. $X\to Y$ ottenuto per **aumento** $\implies$ in $i-1$ passi  $V\to W\in F^A$ (per ipotesi induttiva $V\to W \in F^+$) quindi $X=VZ \land Y=WZ$
>	$$
>	\begin{align}
\forall r\text{ (legale) } t_{1}[X]=t_{2}[X]&\implies t_{1}[VZ]=t_{2}[VZ] \implies \\
&\implies t_{1}[V]=t_{2}[V]\land t_{1}[Z]=t_{2}[Z]
\end{align}
>	$$
>	$$
>	\begin{align} 
>\text{per ipotesi induttiva } V\to W \in F^+ \text{ q}&\text{uindi }t_{1}[W]=t_{2}[W] \implies\\
&\implies t_{1}[WZ]=t_{2}[WZ]\implies \\
&\implies t_{1}[Y]=t_{2}[Y]
\end{align}
>	$$
>
>3. $X\to Y$ ottenuta per **transitività** $\implies X\to Z\in F^A\land Z\to Y\in F^A$
>	$$\forall r \text{ (legale) } t_{1}[X]=t_{2}[X]\implies t_{1}[Z]=t_{2}[Z]\implies t_{1}[Y]=t_{2}[Y]$$
>
>##### Dimostriamo che  $F^+\subseteq F^A$
>$F^+\subseteq F^A$ ovvero $X\to Y\in F^+\implies X\to Y\in F^A$ quindi $X\to Y\in F^A\Leftrightarrow Y\subseteq X^+$
>
>Consideriamo la seguente istanza $r$ di $R$
>![[Pasted image 20241019190252.png|500]]
>L’istanza da solo due tuple, uguali sugli attributi in $X^+$ e diverse in tutti gli altri ($R-X^+$)
>
>Dobbiamo quindi dimostrare che:
>1. $r$ è un’istanza legale di $R$
>2. Avendo dimostrato che $r$ è un’istanza legale allora $X\to Y\in F^+\implies X\to Y\in F^A$
>
>###### $r$ è un’istanza legale di $R$
>Sia $V\to W \in F$
>- se $t_{1}[V]\neq t_{2}[V]$ allora la dipendenza è soddisfatta infatti $(V\cap R-X^+)\neq\varnothing$
>- $$
\begin{flalign}
\text{se }t_{1}[V]= t_{2}[V]&\implies V\subseteq X^+ &&\\
&\xRightarrow{\text{Lemma 1}}X\to V\in F^A \land V\to W\in F  &&\\
&\xRightarrow{\text{Transitività}} X\to W\in F^A &&\\
&\xRightarrow{\text{Lemma 1}} W\subseteq X^+\implies t_{1}[W]=t_{2}[W]
\end{flalign}$$
>
>Quindi $V\to W$ è soddisfatta e avendo considerato una qualsiasi dipendenza in $F$ allora $r$ è un’**istanza legale** di $R$
>###### $X\to Y\in F^+\implies X\to Y\in F^A$
>Supponiamo per assurdo che $X\to Y\in F^+$ e $X\to Y\notin F^A$. Essendo $r$ un’istanza legale le dipendenze di $F^+$ sono soddisfatte anche per $r$.
>Dunque se $t_{1}[X]=t_{2}[X] \text{ allora }t_{1}[Y]=t_{2}[Y]\implies Y\subseteq X^+ \xRightarrow{\text{Lemma 1}} X\to Y\in F^A$
>$\begin{flalign}&& \blacksquare\end{flalign}$

### Nota finale
E’ molto utile notare che la dimostrazione di questo teorema si basa su due collegamenti molto importanti
- il collegamento che esiste tra l’insieme di dipendenze $F^+$ e le istanze legali: se una istanza è legale allora soddisfa anche tutte le dipendenze in $F^+$ e d’altra parte $F^+$ è l’insieme di dipendenze soddisfatte da ogni istanza legale (notare che per verificare se un’istanza è legale basta controllare che soddisfi le dipendenze in $F$)
- il collegamento che esiste tra la chiusura $X^+$di un insieme di attributi $X$ (diamo per scontato che l’insieme di dipendenze di riferimento sia $F$ e otteniamo un pedice) e il sottoinsieme di dipendenze in $F^A$ (che abbiamo appena visto essere uguale a $F^+$) che hanno $X$ come determinante, cioè $Y \subseteq X^+ \Leftrightarrow X\to Y \in F^A$ che equivale a dire che $Y\subseteq X^+\Leftrightarrow X\to Y\in F^+$ e in particolare che $X\to Y$ deve essere soddisfatta da ogni istanza legale

---
## A cosa serve conoscere $F^+$?
Quindi abbiamo un modo per identificare tutte le dipendenze in $F^+$. Sono esattamente le stesse che si possono inserire in $F^A$ partendo da $F$ e applicando gli assiomi di Armstrong e le regole derivate (a costo però di una complessità esponenziale).
Calcolare $F^A$, e quindi $F^+$ richiede tempo esponenziale in $\mid R\mid$. Basta considerare gli assiomi della riflessività e dell’aumento oppure la regola della decomposizione; ogni possibile sottoinsieme di $R$ posta a una o più dipendenze e i possibili sottoinsiemi $R$ sono $2^{\mid R\mid}$ pertanto il calcolo di $\mid F^+\mid$ ha complessità esponenziale in $\mid R\mid$

Parleremo di terza forma normale (3NF), delle sue proprietà e di come ottenerla se lo schema di relazione che abbiamo progettato è carente. Come abbiamo già accennato, il problema nasce dal fatto di aver rappresentato più concetti (oggetti) nella stessa relazione, che la soluzione consiste nel decomporre lo schema in maniera opportuna.
Ovviamente tutte le dipendenze funzionali originarie devono essere soddisfatte anche dalle istanze dei nuovi schemi. **In particolare, devono essere preservate tutte le dipendenze in $F^+$**
---
Created: 2024-11-03
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Utilità della chiusura $X^+$|Utilità della chiusura $X^+$]]
- [[#Come si calcola $X^+$|Come si calcola $X^+$]]
- [[#L’algoritmo è corretto (teorema)|L’algoritmo è corretto (teorema)]]
- [[#Proprietà dell’insieme vuoto|Proprietà dell’insieme vuoto]]
- [[#Esercizi|Esercizi]]
---
## Cosa vogliamo ottenere
Quando si decompone uno schema di relazione $R$ su cui è definito un insieme di dipendenze funzionali $F$, oltre ad ottenere schemi in 3NF occorre
1. **preservare le dipendenze**
2. poter **ricostruire tramite join** tutta e sola l’informazione originaria
Le dipendenze funzionali che si vogliono preservare sono tutte quelle che sono soddisfatte da ogni istanza legale di $R$, cioè le dipendenze funzionali in $F^+$

Quindi si è interessati a calcolare $F^+$ e sappiamo farlo, ma calcolare $F^+$ richiede tempo esponenziale in $\mid R\mid$

>[!info] Ricordiamo
>Se $X\to Y\in F^+$, per le regole della decomposizione e dell’unione, si ha che $X\to Z\in F^+$, per ogni $Z \subseteq Y$; pertanto il calcolo di $\mid F^+\mid$ è esponenziale in $\mid R\mid$

Fortunatamente per i nostri scopi è sufficiente avere un modo per decidere se una dipendenza funzionale $X\to Y$ appartiene ad $F^+$ (cioè alla chiusura di un insieme di dipendenze).

Ciò puà essere fatto calcolando $X^+$ e verificando se $Y\subseteq X^+$. Infatti ricordiamo il lemma: $X\to Y\in F^A$ se e solo se $Y\subseteq X^+$ e il teorema che dimostra che $F^A=F^+$

---
## Utilità della chiusura $X^+$
Vedremo che il calcolo di $X^+$ è utile in diversi casi:
- verificare le condizioni perché un insieme di attributi sia chiave di uno schema
- verificare se una decomposizione preserva le dipendenze funzionali dello schema originario
- altro ancora…

---
## Come si calcola $X^+$
Per il calcolo della chiusura dell’insieme di attributi $X$, denotata con $X^+$, possiamo usare il seguente algoritmo

**Input** → uno schema di relazione $R$, e un insieme $F$ di dipendenze funzionali su $R$, un sottoinsieme $X$ di $R$
**Output** → la chiusura di $X$ rispetto ad $F$ (restituita nella variabile `Z`)

$$
\begin{flalign}
&\text{begin}\\
&Z:=X\\
&S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}\\
&\text{while } S\not\subset Z\\
&\qquad\text{do}\\
&\qquad\text{begin}\\
&\qquad\qquad Z:=Z\cup S\\
&\qquad\qquad S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}\\
&\qquad\text{end}\\
&\text{end}\\
\end{flalign}
$$

Si inseriscono in $S$ i singoli attributi che compongono le parti destre di dipendenze in $F$ la cui parte destra è contenuta in $Z$ (in pratica decomponendo le parti destre). All’inizio $Z$ è proprio $X$, quindi inseriamo gli attributi che sono determinati funzionalmente da $X$; una volta che questi sono entrati in $Z$, da questi ne aggiungiamo altri (per transitività).
Possiamo “numerare” gli insiemi $Z$ successivi ($Z^{(i)}$ è l’insieme ottenuto dopo la i-esima iterazione del while)

All’iterazione $i+1$ aggiungiamo in $S$ i songoli attribut che compongono le parti destre di dipendenze in $F$ la cui parte sinistra è contenuta in $Z^{(i)}$ cioè $S:=\{A \mid Y\to V\in F,\,\, A\in V,\,\, Y\subseteq Z\}$. Alla fine di ogni iterazione aggiungiamo qualcosa a $Z$, ma non eliminiamo mai nessun attributo

L’algoritmo si ferma quando il nuovo insieme $S$ che otteniamo è (già) contenuto nell’insieme $Z$, cioè quando non possiamo aggiungere nuovi attributi alla chiusura transitiva di $X$

>[!example]
>$$F=\{AB\to C, B\to D, AD\to E, CE\to H\}$$
>$$R=A,B,C,D,E,H,L$$
> Vogliamo calcolare la chiusura di $AB$
> $Z=A,B$
> $S=\{C,D\}$
> inseirco c perché $AB\to C\in F$, per inserire $D$ ho $AB\to B \text{ (riflessiva)}+B\to D(\in F)=AB\to D \text{ (transitiva)}$
> 
> $S$ ha qualcosa in più?
> $Z=\{A,B,C,D\}$
> $S=\{C,D,E\}$ per inserire $E$ ho $AB\to B+B\to D+AB\to AD+AD\to E=AB\to E$
> 
> $S$ ha qualcosa in più?
> $Z=\{A,B,C,D, E\}$
> $S=\{C,D,E,H\}$ per inserire $H$ ho $AB\to C+AB\to AD+AD\to E+AB\to E+AB\to E+AB\to CE+CE\to H=AB\to H$
> $S$ ha qualcosa in più?
> $Z=\{A,B,C,D, E, H\}$
> $S=\{C,D,E,H\}$
> 
> $S$ ha qualcosa in più?
> STOP

---
		## L’algoritmo è corretto (teorema)
L’algoritmo per il calcolo di $X^+$ calcola correttamente la chiusura di un insieme di attributi $X$ rispetto ad un insieme $F$ di dipendenze funzionali.

>[!info] Dimostrazione
>Indichiamo con $Z^{(0)}$ il valore iniziale di $Z \,(Z^{(0)}=X)$ e con $Z^{(i)}$ ed $S^{(i)}$ con $i\geq 1$, i valori di $Z$ ed $S$ dopo l’i-esima esecuzione del corpo del ciclo.
>E’ facile vedere che $Z^{(i)}\subseteq Z^{(i+1)}$, per ogni $i$
>>[!hint] Ricordiamo
>>In $Z^{(i)}$ ci sono attributi aggiunti a $Z$ fino all’i-esmia iterazione.
>>
>>Alla fine di ogni iterazione aggiungiamo qualcosa a $Z$, ma non eliminiamo mai alcun attributo
>
>Sia $j$ tale che $S^{(j)}\subseteq Z^{(j)}$ (cioè $Z^{(j)}$ è il valore di $Z$ quando l’algoritmo termina); proveremo che: $\mathbf{A\in Z^{(j)}\Leftrightarrow A\in X^+}$
>
>##### Parte $\Rightarrow$
>Mostreremo per induzione su $i$ che $Z^{(i)}\subseteq X^+$, per ogni $i$ (e quindi, in particolare $Z^{(j)}\subseteq X^+$)
>- Base dell’induzione ($i=0$): poiché $Z^{(0)}=X$ e $X\subseteq X^+$, si ha $Z^{(0)}\subseteq X^+$
>- Ipotesi induttiva ($i>0$): $Z^{(i-1)}\subseteq X^+\overset{\text{Lemma 1}}{\Longrightarrow}X\to Z^{(i-1)}\in F^A$
>- Passo induttivo: $Z^{(i)}$
>
>Sia $A$ un attributo in $Z^{(i)}-Z^{(i-1)}$ allora deve esistere una dipendenza $Y\to V\in F$ tale che $Y\subseteq Z^{(i-1)}$ e $A\in V$.
>
>Poiché $Y\subseteq Z^{(i-1)}$, per l’ipotesi induttiva si ha che $Y\subseteq X^+\Rightarrow X\to Y\in F^A$
>$\begin{aligned}X\to Y\in F^A\overset{\text{trans}}{\Longrightarrow}&\,\, X\to V\in F^A{\Longrightarrow}\\\overset{A\in V}{\Longrightarrow}& \,\,X\to A\in F^A\Longrightarrow A\in X^+\end{aligned}$
>
>##### Parte $\Leftarrow$
>Devo quindi dimostrare che $A\in X^+\Rightarrow A\in Z^{(j)}$
>
>Poiché $A\in X^+$, si ha $X\to A\in F^A=F^+$; pertanto $X\to A$ deve essere soddisfatta per ogni istanza legale di $R$. Si consideri la seguente istanza $r$ di $R$
>![[Screenshot 2024-11-03 alle 01.46.22.png|500]]
>
>Dobbiamo quindi dimostrare che:
>1. $r$ è un’istanza legale di $R$
>2. $A\in X^+\Rightarrow A\in Z^{(j)}$
>###### $r$ è un’istanza legale di $R$
>Supponiamo per assurdo che la dipendenza $V\to W\in F$ non è soddisfatta.
>Avremmo quindi $t_{1}[V]=t_{2}[V]\land t_{1}[W]\neq t_{2}[W]$; il che vuol dire che $V\subseteq Z^{(j)}$ e $W\cap(R-Z^{j})\neq \varnothing$
>
>Ma ciò non è possibile in quanto non sarebbe l’ultima iterazione ($Z^{(j)}$) in quanto manca $W$ che invece ci dovrebbe essere in quanto $V\to W\in F$. Se non li ho ancora aggiunti $Z^{(j)}$ non è la versione finale ma questo è in contraddizione con la nostra costruzione dell’istanza
>
>Quindi $W\subseteq Z^{(j)}\Rightarrow t_{1}[W]=t_{2}[W]$ terminando così la dimostrazione che questa istanza è legale
>###### $A\in X^+\Rightarrow A\in Z^{(j)}$
>Come detto precedentemente $X\to A\in F^+$ ed essendo questa un’istanza legale di $R$ anche qui deve essere soddisfatta.
>Sapendo che $X= Z^{(0)}\subseteq Z^{(j)}$ allora le due tuple devono essere anche uguali su $A$, quindi $A\in Z^{(j)}$
>$\begin{flalign}&& \blacksquare\end{flalign}$

---
## Proprietà dell’insieme vuoto

>[!warning]
>Prima di tutto va sottolineato che la notazione $\{\varnothing\}$ indica l’insieme che contiene l’insieme vuoto (insieme di insiemi) e non va pertanto confusa con il semplice insieme vuoto $\varnothing$

- L’insieme vuoto è un **sottoinsieme** di ogni insieme $A$
	$\forall A : \varnothing\subseteq A$
- L’**unione** di un qualunque insieme $A$ con l’insieme vuoto è $A$
	$\forall A:A\cup \varnothing=A$
- L’**intersezione** di un qualunque insieme $A$ con l’insieme vuoto è l’insieme vuoto
	$\forall A:A\cap \varnothing=\varnothing$
- Il **prodotto cartesiano** di un qualunque insieme $A$ con l’insieme vuoto è l’insieme vuoto
	$\forall A:A\times \varnothing=\varnothing$
- L’unico sottoinsieme dell’insieme vuoto è l’insieme vuoto stesso
- Il numero di elementi dell’insieme vuoto (vale a dire la sua **cardinalità**) è **zero**; l’insieme vuoto è quindi finito: $\mid \varnothing\mid=0$

---
## Esercizi
>[!example] Esercizio 1
>$$R=(A,B,C,D,E,H)$$
>$$F=\{AB\to CD, EH\to D, D\to H\}$$
>Calcolare le chiusure degli insiemi $A$, $D$ e $AB$
>
>$A^+=\{A\}$
>$D^+=\{D,H\}$
>$AB^+=\{A,B,C,D,H\}$

>[!example] Esercizio 2
>$$R=(A,B,C,D,E,H,I)$$
>$$F=\{A\to E,AB\to CD,EH\to I,D\to H\}$$
>Calcolare la chiusura dell’insieme $AB$
>
>$AB^+=\{A,B,C,D,H,E,I\}$


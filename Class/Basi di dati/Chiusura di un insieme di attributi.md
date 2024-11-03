---
Created: 2024-11-03
Class: "[[Basi di dati]]"
Related: 
Completed:
---
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


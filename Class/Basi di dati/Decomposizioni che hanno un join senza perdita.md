---
Created: 2024-11-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Quando si decompone uno schema occorre tenere presente (oltre al dover mantenere tutte le dipendenze originali), deve permettere di **ricostruire mediante il join naturale** ogni **istanza legale dello schema originario** (senza aggiunta di tuple estranee)
[[Terza forma normale#La 3NF non basta#Esempi|Qui]] qualche esempio in cui ciò non accade

---
## Definizione
Se si decompone uno schema di relazione $R$ si vuole che la decomposizione $\{R_{1},R_{2},\dots,R_{k}\}$ ottenuta sia tale che ogni istanza legale $r$ di $R$ sia ricostruibile mediante join naturale ($\bowtie$) da un’istanza legale $\{r_{1},r_{2},\dots,r_{k}\}$ dello schema decomposto $\{R_{1},R_{2},\dots,R_{k}\}$

>[!info] Definizione
>Sia $R$ uno schema di relazione. Una decomposizione $\rho=\{R_{1},R_{2},\dots,R_{k}\}$ di $R$ ha un **join senza perdita** se per ogni istanza legale $r$ di $R$ si ha $r=\pi_{R_{1}}(r)\bowtie \pi_{R_{2}}(r)\bowtie\dots \bowtie \pi_{R_{k}}(r)$

---
## Teorema
Sia $R$ uno schema di relazione e $\rho=\{R_{1},R_{2},\dots ,R_{k}\}$ una decomposizione di $R$. Per ogni istanza legale $r$ di $R$, indicato con $m_{\rho}(r)=\pi_{R1}(r)\bowtie \pi_{R2}(r)\bowtie\dots \bowtie \pi_{Rk}(r)$ si ha:
- $r \subseteq m_{\rho}(r)$
- $\pi_{R_{i}}(m_{\rho}(r))=\pi_{R_{i}}(r)$
- $m_{\rho}(m_{\rho}(r))=m_{\rho}(r)$
### Digressione
Consideriamo la solita istanza legale di $R=ABC$ con l’insieme di dipendenze funzionali $F=\{A\to B,C\to B\}$ (non è in 3NF - $B$ non è contenuto in una chiave)

![[Pasted image 20241117185711.png|250]]

In base alle possibili decomposizioni dello schema, questa istanza si decompone in
![[Pasted image 20241117185753.png|350]]
La prima è ottenuta proiettando l’istanza originale su $AB$. La seconda è ottenuta proiettando l’istanza originale su $BC$.

>[!note]
>Notare l’eliminazione del duplicato. Forse questa eliminazione ci farà perdere tuple originali? No

Dovrebbe essere possibile ricostruire l’istanza di partenza esattamente tramite join invece se si effettua il join delle due istanze legali risultanti dalla decomposizione si ottiene

![[Pasted image 20241117190130.png]]

Occorre garantire che il join delle istanze risultati dalla decomposizione non riveli *perdita di informazioni*

![[Screenshot 2024-11-17 alle 19.04.41.png]]

![[Screenshot 2024-11-17 alle 19.15.14.png]]

---
## Algoritmo di verifica
Il seguente algoritmo permette di verificare se una decomposizione data ha un join senza perdita in tempo polinomiale

$$
\begin{align}
\mathbf{Input}\quad&\text{uno schema di relazione }R\text{, un insieme }F\text{ di dipendenze funzionali su }R\text{, una}\\&\text{ decomposizione }\rho=\{R_{1},R_{2},\dots,R_{k}\}\text{ di }R \\
\mathbf{Output}\quad&\text{decide se }\rho \text{ ha un join senza perdita}
\end{align}
$$
$$
\begin{align}
&\mathbf{begin} \\
&\text{Costruisci una tabella }r\text{ nel modo seguente:} \\
&r\text{ ha } \mid R\mid \text{colonne e }\mid \rho\mid \text{ righe} \\
&\text{all'incrocio dell'i-esima riga e della j-esima colonna metti} \\
&\text{il simbolo }a_{j}\text{ se l'attributo }A_{j}\in R_{i} \\
&\text{il simbolo }b_{ij}\text{ altrimenti} \\
&\mathbf{repeat} \\
&\mathbf{for\,\,every\,\, X\to Y\in F} \\
&\qquad\mathbf{do\,\,if} \text{ ci sono due tuple }t_{1}\text{ e }t_{2}\text{ in }r\text{ tali che }t_{1}[X]=t_{2}[X] \text{ e }t_{1}[Y]\neq t_{2}[Y] \\
&\qquad\qquad\mathbf{then\,\,for\,\,every\,\,attribute\,\,A_{j}\in Y} \\
&\qquad\qquad\qquad\mathbf{do\,\,if\,\,}t_{1}[A_{j}]='{a_{j}}' \\
&\qquad\qquad\qquad\qquad\mathbf{then\,\,}t_{2}[A_{j}]:=t_{1}[A_{j}] \\
&\qquad\qquad\qquad\qquad\mathbf{else\,\,}t_{1}[A_{j}]:=t_{2}[A_{j}] \\
&\mathbf{until\,\,}r\text{ ha una riga con tutte 'a' }\mathbf{or\,\,}r\text{ non è cambiato} \\
&\mathbf{if\,\,}r\text{ ha una riga con tutte 'a'} \\
&\qquad\mathbf{then\,\,}\rho \text{ ha un join senza perdita} \\
&\qquad\mathbf{else\,\,}\rho \text{ non ha un join senza perdita}
\end{align}
$$

### Esempi
>[!example]
>$$R=(A,B,C,D,E)$$
>$$F=\{C\to D,AB\to E,D\to B\}$$
>$$\rho=\{AC,ADE,CDE,AD,B\}$$
>Verificare se la decomposizione $\rho$ ha un join senza perdita
>
>Cominciamo a costruire la relativa tabella
>![[Pasted image 20241117212115.png|440]]
>
>>[!info]
>>Per chiarezza applichiamo le dipendenze funzionali nell’ordine e vediamo i cambiamenti che vengono effettuati sulla tabella (ricordiamo che ogni cambiamento corrisponde a fare in modo che venga soddisfatta una dipendenza funzionale, per ottenere alla fine dell’algoritmo una tabella che rappresenta un’istanza legale dello schema)
>>Indicheremo col simbolo $\to$ le modifiche ai valori della tabella e con un apice l’ordine delle sostituzioni quando opportuno
>
>![[Pasted image 20241117212420.png|500]]
>- $C\to D$
>	- la prima e la terza riga coincidono sull’attributo $C=a3$, quindi cambiamo $b14$ in $a4$ in modo che la dipendenza funzionale sia soddisfatta (solo le righe che hanno valori uguali in $C$ devono avere valori uguali in $D$)
>- $AB\to E$
>	- non viene utilizzata in questo passo: la dipendenza funzionale è già soddisfatta, in quanto non ci sono (ancora) tuple uguali su $AB$ e diverse su $E$, quindi non devono essere effettuati cambiamenti
>- $D\to B$
>	- nelle prime quattro righe $D=a4$, quindi cambiamo $b22$ in $b12$, $b32$ in $b12$, $b42$ in $b12$ (potevamo scegliere una diversa sostituzione delle $b$, purché le rendesse tutte uguali)
>
>Abbiamo completato la prima iterazione del for e la tabella è stata modificata quindi continuiamo
>![[Pasted image 20241117213039.png|500]]
>- $C\to D$
>	- non viene utilizzata in questo passo: la dipendenza funzionale è già soddisfatta in quanto non ci sono tuple uguali su $C$ e diverse su $D$
>- $AB\to E$
>	- la prima, la seconda e la quarta riga coincidono sugli attributi $AB=<a1,b12>$, quindi cambiamo $b15$ in $a5$ e $b45$ in $a5$ in modo che la dipendenza funzionale sia soddisfatta (se le righe hanno valori uguali in $AB$, devono avere valori uguali in $E$)
>- $D\to B$
>	- non vine utilizzata in questo passo: la dipendenza funzionale è già soddisfatta in quanto non ci sono tuple uguali su $A$ e diverse su $B$
>
>Abbiamo completato la seconda iterazione del for e la tabella è stata modificata quindi continuiamo
>![[Pasted image 20241117213510.png|500]]
>- $C\to D$
>	- non viene utilizzata in questo passo
>- $AB\to E$
>	- non viene utilizzata in questo passo
>- $D\to B$
>	- non vine utilizzata in questo passo
>
>La tabella non cambia più e quindi l’algoritmo termina. Ora occorre verificare la presenza della tupla con tutte $a$
>Poiché non c’è una riga con tutte $a$, il join non è senza perdita

>[!example]
>$$R=(A,B,C,D,E,H,I)$$
>$$F=\{A\to B,B\to AE,DI\to B,D\to HI,HI\to C,C\to A\}$$
>$$\rho=\{ACD,BDEH,CHI\}$$
>Verificare se la decomposizione $\rho$ ha un join senza perdita
>
>Cominciamo a costruire la tabella
>![[Pasted image 20241117213918.png|440]]
>
>![[Pasted image 20241117213944.png|550]]
>- $A\to B$
>	- non si applica a questa iterazione
>- $B\to AE$
>	- non si applica a questa iterazione
>- $DI\to B$
>	- ci sono due tuple uguali su $D$ ma non su $I$, non si applica a questa iterazione
>- $D\to HI$
>	- la prima e la seconda riga coincidono sull’attributo $D=a4$, quindi cambiamo $H$ e $I$ ma separatamente $b16\to a6$ mentre $b27\to b17$
>- $HI \to C$
>	- ora abbiamo due tuple uguali su $HI$ (la prima e la seconda entrambe con valori $<a6, b17>$) quindi modifichiamo i valori della $C$ nelle stesse tuple - $b23\to a3$
>- $C\to A$
>	- le tuple sono tutte uguali su $C$, quindi le facciamo diventare uguali su $A$, e poiché abbiamo la prima con valore $a$, diventano tutte $a$ ($b21\to a1$, $b31\to a1$)
>
>Abbiamo completato la prima iterazione del ciclo for e la tabella è stata modificata quindi continuiamo
>![[Pasted image 20241117214538.png|550]]
>- $A\to B$
>	- le tuple sono tutte uguali su $A$, quindi le facciamo diventare uguali su $B$ e poiché abbiamo la seconda con valore $a$, diventano tutte $a$ ($b12\to a2$, $b32\to a2$)
>- $B\to AE$
>	- ora tutte le tuple sono uguali su $B$, quindi devono diventare uguali anche su $AE$; su $A$ sono già uguali, la seconda tupla ha una $a$ sull’attributo $E$, quindi diventeranno tutte $a$ ($b15\to a5$, $b35\to a5$)
>- $DI\to B$
>	- la prima e la seconda tupla sono uguali su $DI=<a4,b17>$, quindi devono diventare uguali su $B$ ma lo sono già
>- $D\to HI$
>	- già soddisfatta, nulla da modificare
>- $HI \to C$
>	- già soddisfatta, nulla da modificare
>- $C\to A$
>	- già soddisfatta, nulla da modificare
>
>Abbiamo completato la seconda iterazione del ciclo for e la tabella è stata modificata quindi continuiamo
>![[Pasted image 20241117215041.png|550]]
>- $A\to B$
>	- già soddisfatta, nulla da modificare
>- $B\to AE$
>	- già soddisfatta, nulla da modificare
>- $DI\to B$
>	- già soddisfatta, nulla da modificare
>- $D\to HI$
>	- già soddisfatta, nulla da modificare
>- $HI \to C$
>	- già soddisfatta, nulla da modificare
>- $C\to A$
>	- già soddisfatta, nulla da modificare
>
>Abbiamo completato l’iterazione del for e la tabella non è stata modificata, quindi l’algoritmo termina. Occorre verificare la presenza della tupla con tutte $a$
>Poiché non c’è una rica con tutte $a$, il join non è senza perdita



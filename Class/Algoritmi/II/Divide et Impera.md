---
Created: 2025-04-17
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Il problema della selezione
Data una lista $A$ di $n$ interi distinti, ed un intero $k$, con $1\leq k\leq n$, vogliamo sapere quale elemento occuperebbe la posizione $k$ se il vettore venisse ordinato

>[!hint] Casi particolari
>- per $k=1$ avremo il *minimo* di $A$
>- per $k=n$ avremo il *massimo* di $A$
>- per $k=\left\lceil  \frac{n}{2}  \right\rceil$ avremo il *mediano* di $A$

Un semplice algoritmo di complessità $\Theta (n\log n)$ è il seguente:
```python
def selezione1(A, k):
	A.sort
	return A[k-1]
```

Se $k=1$ o $k=n$ il problema si riduce alla ricerca del minimo o del massimo e questi casi sono risolvibili in tempo $\Theta(n)$
Vedremo che utilizzano la tecnica del **divide et impera**, il problema può risolversi in $\Theta(n)$ anche nel caso generale. In altre parole dimostreremo che il problema della selezione è computazionalmente più semplice di quello dell’ordinamento

### Approccio basato sul divide et impera
Un possibile approccio è:
- scegli nella lista $A$ l’elemento in posizione $A[0]$ (che chiameremo **pivot**)
- a partire da $A$ costruisci le liste $A_{1}$ ed $A_{2}$, la prima contenente gli elementi di $A$ minori del pivot e la seconda gli elementi di $A$ maggiori del pivot
- l’elemento di rango $k$ si trova:
	- nel vettore $A_{1}$ se $| A_{1}|\geq k$
	- è il pivot $x$ se $| A_{1}|=k-1$
	- nel vettore $A_{2}$ è l’elemento di rango $k-|A_{1}|-1$ se $|A_{1}|<k-1$

>[!info]
>Dopo aver costruito la lista $A$ di $n$ elementi le due liste $A_{1}$ e $A_{2}$, grazie al test sulla cardinalità di $A_{1}$ il problema della selezione dell’elemento di rango $k$ o risulta risolto o viene ricondotto alla selezione di un elemento in una lista con meno di $n$ elementi

```python
def selezione2(A, k):
	if len(A) == 1:
		return A[0]
	pivot = A[0]
	A1, A2 = [], []
	for i in range(1,len(A)):
		if A[i]<pivot:
			A1.append(A[i])
		else:
			A2.append(A[i])
	if len(A1)>=k:
		return slezione2(A1,k)
	elif len(A1) == k-1:
		return pivot
	return selezione2(A2, k-len(A1)-1)
```

La procedura che tripartisce la lista in $A_{1}$, $A[0]$ e $A_{2}$ può restituire una partizione massimamente sbilanciata in cui si ha ad esempio $|A_{1}|=1$ e $|A_{2}|=n-1$, questo accade quando il pivot risulta l’elemento minimo nella lista.
Qualora questo evento sfortunato si ripetesse sistematicamente nel corso delle varie partizioni eseguite dall’algoritmo (questo può accadere quando cerco il massimo in una lista ordinata), allora la complessità dell’algoritmo al caso pessimo viene catturata dalla seguente equazione:
$$
T(n)=T(n-1)+\Theta(n)=T(n)=\Theta(n^2)
$$

In generale la complessità superiore alla procedura è catturata dalla ricorrenza:
$$
T(n)=T(m)+\Theta(n)
$$
dove $m=\text{max}\{|A_{1}|,|A_{2}|\}$

Se avessimo una regola di scelta del pivot in grado di garantire una partizione bilanciata, ossia:
$$
m=\text{max}\{|A_{1}|,|A_{2}|\}\approx \frac{n}{2}
$$
allora per la complessità $T(n)$ dell’algoritmo avremmo:
$$
T(n)=T\left( \frac{n}{2} \right)+\Theta(n)=\Theta(n)
$$

Chiedere però che la partizione sia perfettamente bilanciata è forse chiedere troppo, potremmo allora accontentarci di chiedere che la scelta del primo garantisca partizioni non troppo sbilanciate come ad esempio quelle per cui:
$$
m=\text{max}\{|A_{1}|,|A_{2}|\}\approx \frac{3}{4}n
$$
In questo caso si ha:
$$
T(n)\leq T\left( \frac{3}{4} n\right)+\Theta(n)=\Theta(n)
$$
In generale finché $m$ è una frazione di $n$ (anche piuttosto vicina ad $n$ come ad esempio $\frac{99}{100}n$) la ricorrenza dà sempre $T(n)=\Theta(n)$

---
## Scelta del pivot in modo equiprobabile
Una possibile idea per risolvere questo problema è quindi quella di scegliere il pivot $p$ a caso in modo equiprobabile tra gli elementi della lista

Anche se la scelta “casuale” non produce necessariamente una partizione bilanciata, quanto visto ci fa intuire che la complessità rimane lineare in $n$

```python
def selezione2R(A,k):
	if len(A)==1:
		return A[0]
	pivot = A[randint(0, len(A)-1)]
	A1, A2 = [], []
	for x in A:
		if x<pivot:
			A1.append(x)
		elif x>pivot:
			A2.append(x)
	if len(A1)>=k:
		return selezione2R(A1,k)
	elif len(A1)==k-1:
		return pivot
	return selezione2R(A2, k-len(A1)-1)
```

> [!info] Analisi formale del caso medio
> Con la randomizzazione introdotta per la scelta del pivot possiamo assumere che uno qualunque degli elementi del vettore, con uguale probabilità $\frac{1}{n}$, diventi pivot e, poiché la scelta dell’elemento di rango $k$ produce $|A_{1}|=k-1$ e $|A_{2}|=n-k$, per il tempo atteso dell’algoritmo va studiata la ricorrenza:
> $$
> T(n)\leq \frac{1}{n}\sum^n_{k=1}T\Big(\text{max}\big\{T(k-1),T(n-k)\big\}\Big)+\Theta(n)\leq \frac{1}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }2T(k)+\Theta(n)
> $$
> possiamo dimostrare che per questa ricorrenza vale $T(n)=O(n)$ col metodo di sostituzione
> $$
> T(n)=
> \begin{cases}
> \frac{1}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }2T(k)+a\cdot n&\text{se }n\geq 3 \\
> b&\text{altrimenti}
> \end{cases}
> $$
> Dimostriamo $T(n)<cn$ per una qualunque $c>0$ costante
> Per $n\leq 3$ abbiamo $T(n)\leq b\leq 3c$ che è vera ad esempio per $c\geq b$
> 
> Sfruttando l’ipotesi induttiva $T(k)\leq c\cdot k$ per $k<n$ abbiamo
> $$
> T(n)\leq \frac{2c}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }k+a\cdot n
> $$
> da cui ricaviamo
> $$
> \begin{align}
> T(n)&\leq \frac{2c}{n}\left( \sum^{n-1}_{k=1}k-\sum^{\lfloor n/2 \rfloor -1}_{k=1}k \right)+a\cdot n\leq \frac{2c}{n}\left( \frac{n(n-1)}{2}-\frac{\left( \frac{n}{2}-1 \right)\left( \frac{n}{2}-2 \right)}{2} \right)+a\cdot n \leq\\
> &\leq \frac{c}{n}\left( \frac{3n^2}{4}+\frac{n}{2}-2 \right)+a\cdot n\leq \frac{3cn}{4}+\frac{c}{2}+a\cdot n=cn-\left( \frac{cn}{4}-\frac{c}{2}-a\cdot n \right)\leq cn
> \end{align}
> $$
> dove l’ultima diseguaglianza segue prendendo $c$ in modo che $\left( \frac{cn}{4}-\frac{c}{2}-a\cdot n \right)\leq 0$; basta ad esempio prendere $c\geq 8a$

L’analisi rigorosa appena fatta dimostra che, se la scelta del pivot avviene in modo equiprobabile a caso tra i vari elementi della lista $A$, il tempo di calcolo dell’algoritmo risulta con altra probabilità lineare in $n$.
Ovviamente nel caso peggiore, quando nelle varie partizioni che si succedono nell’interazione dell’algoritmo si verifica che il perno scelto a caso risulta sempre vicino al massimo o al minimo della lista, la complessità rimane $O(n^2)$. Però questo accade con probabilità molto piccola

---
## Soluzione in $O(n)$
Vedremo ora un algoritmo deterministico che garantisce complessità $O(n)$ anche nel caso pessimo

Abbiamo visto che riuscire a selezionare un pivot in grado di garantire che nessuna delle due sottoliste $A_{1}$ e $A_{2}$ abbia più di $c\cdot n$ elementi, per una qualche costante $0<c<1$, avrebbe come conseguenza una complessità di calcolo $O(n)$

Descriviamo ora un metodo (noto come il *mediano dei mediani*) per selezionare un pivot che garantisce di produrre sempre due sottoliste $A_{1}$ ed $A_{2}$ ciascuna delle quali non ha più di $\frac{3}{4}n$ elementi

### Algoritmo per la selezione
Iniziamo dividendo l’insieme $A$, contentente $n$ elementi, in gruppi da $5$ elementi ciascuno; l’ultimo gruppo però potrebbe avere meno di $5$ elementi, quindi consideriamo solo i primi $\left\lfloor  \frac{n}{5}  \right\rfloor$ gruppi, ciascuno composto esattamente da $5$ elementi
Quindi bisogna trovare il mediano all’interno di ciascuno di questi $\left\lfloor  \frac{n}{5}  \right\rfloor$ gruppi e infine calcoliamo il mediano $p$ dei mediani ottenuti.

Useremo $p$ come elemento pivot per l’insieme $A$

>[!example] Scelta del perno
>![[Pasted image 20250419214511.png]]

>[!info] Proprietà
>Se la lista $A$ contiene almeno $120$ elementi e il perno $p$ con cui partizionarla viene scelto in base alla regola appena descritta si può essere sicuri che la dimensione di ciascuna delle due sottoliste $A_{1}$ e $A_{2}$ ottenute sarà limitata da $\frac{3}{4}n$
>
>>[!done] Prova
>>Il perno scelto $p$ ha la proprietà di trovarsi in posizione $\left\lceil  \frac{n}{10}  \right\rceil$ nella lista degli $\left\lfloor  \frac{n}{5}  \right\rfloor$ mediani selezionati in $A$.
>>Ci sono dunque $\left\lceil  \frac{n}{10}  \right\rceil-1$ mediani di valore inferiore a $p$ e $\left\lceil  \frac{n}{5}  \right\rceil-\left\lfloor  \frac{n}{10}  \right\rfloor$ mediani di valore superiore a $p$
>>
>>##### Prova che $|A_{2}|< \frac{3}{4}n$
>>Considera i $\left\lceil  \frac{n}{10}  \right\rceil-1$ mediani di valore inferiore a $p$. Ognuno di questi mediani appartiene ad un gruppo di $5$ elementi in $n$. Ci sono dunque in $A$ altri $2$ elementi inferiori a $p$ per ogni mediano
>>
>>In totale abbiamo
>>$$3\left( \left\lceil  \frac{n}{10}  \right\rceil -1 \right)\geq 3 \frac{n}{10}-3$$
>>elementi di $A$ che finiranno in $A_{1}$
>>
>>Abbiamo dunque:
>>$$|A_{2}|\leq n-\left( 3 \frac{n}{10} -3\right)=\frac{7}{10}n+3\leq \frac{3}{4}n$$
>>dove l’ultima diseguaglianza segue dal fatto che $n\geq 120$
>>
>>##### Prova che $|A_{1}|< \frac{3}{4}n$
>>Ci sono invece:
>>$$\left\lfloor  \frac{n}{5}  \right\rfloor -\left\lceil  \frac{n}{10}  \right\rceil \geq\left( \frac{n}{5}-1 \right)-\left( \frac{n}{10}+1 \right)=\frac{n}{10}-2$$
>>mediani di valore superiore a $p$
>>
>>Ognuno di questi mediani appartiene ad un gruppo di $5$ elementi in $A$. Ci sono dunque in $A$ altri $2$ elementi superiori a $p$ per ogni mediano.
>>
>>Sostituendo abbiamo in totale almeno
>>$$3 \frac{n}{10}-6$$
>>elementi di $A$ che finiranno in $A_{2}$
>>
>>Abbiamo dunque:
>>$$|A_{2}|\leq n-\left( 3 \frac{n}{10} -6\right)=\frac{7}{10}n+6\leq \frac{3}{4}n$$
>>dove l’ultima diseguaglianza segue dal fatto che $n\geq 120$

```python
from math import ceil

def selezione(A,k):
	if len(A)<=120: # niente sondaggio, costo costante 120 log 120
		A.sort()
		return A[k-1]
	
	# inizializza B con i mediani dei len(A)//5 gruppi di 5 elementi di A
	B = [sorted(A[5*1 : 5*i+5])[2] for i in range(len(A)//5)] #
	# individua il pivot p con la revola del mediano dei mediani
	pivot = selezione(B, ceil(len(A)/10))
	
	A1, A2 = [], []
	for x in A:
		if x<pivot:
			A1.append(x)
		elif x>pivot:
			A2.append(x)
	if len(A1)>=k:
		return selezione(A1,k)
	elif len(A1)==k-1:
		return pivot
	return selezione(A2, k-len(A1)-1)
```

>[!hint]
>- ordinare $120$ elementi richiede tempo $O(1)$
>- ordinare una lista di $n$ elementi in gruppetti di $5$ richiede $\Theta\left( \frac{n}{5} \right)=\Theta(n)$ (infatti ordinare una lista di $5$ elementi richiede tempo costante)
>- selezionare i mediani dei mediani di gruppi da $5$ da una lista in cui gli elementi sono stati ordinati in gruppetti  
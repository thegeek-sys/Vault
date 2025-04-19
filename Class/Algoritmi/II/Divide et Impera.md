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

### Scelta del pivot in modo equiprobabile
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

#### Analisi formale del caso medio
Con la randomizzazione introdotta per la scelta del pivot possiamo assumere che uno qualunque degli elementi del vettore, con uguale probabilità $\frac{1}{n}$, diventi pivot e, poiché la scelta dell’elemento di rango $k$ produce $|A_{1}|=k-1$ e $|A_{2}|=n-k$, per il tempo atteso dell’algoritmo va studiata la ricorrenza:
$$
T(n)\leq \frac{1}{n}\sum^n_{k=1}T\Big(\text{max}\big\{T(k-1),T(n-k)\big\}\Big)+\Theta(n)\leq \frac{1}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }2T(k)+\Theta(n)
$$
possiamo dimostrare che per questa ricorrenza vale $T(n)=O(n)$ col metodo di sostituzione
$$
T(n)=
\begin{cases}
\frac{1}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }2T(k)+a\cdot n&\text{se }n\geq 3 \\
b&\text{altrimenti}
\end{cases}
$$
Dimostriamo $T(n)<cn$ per una qualunque $c>0$ costante
Per $n\leq 3$ abbiamo $T(n)\leq b\leq 3c$ che è vera ad esempio per $c\geq b$

Sfruttando l’ipotesi induttiva $T(k)\leq c\cdot k$ per $k<n$ abbiamo
$$
T(n)\leq \frac{2c}{n}\sum^{n-1}_{k=\left\lfloor  \frac{n}{2}  \right\rfloor }k+a\cdot n
$$
da cui ricaviamo
$$
\begin{align}
T(n)&\leq \frac{2c}{n}\left( \sum^{n-1}_{k=1}k-\sum^{\lfloor n/2 \rfloor -1}_{k=1}k \right)+a\cdot n\leq \frac{2c}{n}\left( \frac{n(n-1)}{2}-\frac{\left( \frac{n}{2}-1 \right)\left( \frac{n}{2}-2 \right)}{2} \right)+a\cdot n \leq\\
&\leq \frac{c}{n}\left( \frac{3n^2}{4}+\frac{n}{2}-2 \right)+a\cdot n\leq \frac{3cn}{4}+\frac{c}{2}+a\cdot n=cn-\left( \frac{cn}{4}-\frac{c}{2}-a\cdot n \right)\leq cn
\end{align}
$$
dove l’ultima diseguaglianza seguen prendendo $c$ in modo che $\left( \frac{cn}{4}-\frac{c}{2}-a\cdot n \right)\leq 0$; basta ad esempio prendere $c\geq 8a$

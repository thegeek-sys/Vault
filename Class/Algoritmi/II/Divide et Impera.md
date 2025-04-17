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

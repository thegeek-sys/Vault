---
Created: 2025-05-15
Class: "[[Algoritmi]]"
Related:
---
---
## Esercizi

>[!question] Progettare un algoritmo che prendere come parametro un intero $n$ e stampa tutte le stringhe binarie lunghe $n$
>Ad esempio per $n=3$ l’algoritmo deve stampare $2^3=8$ stringhe:
>```
>000, 001, 010, 100, 011, 101, 110, 111
>```
>![[Pasted image 20250515001541.png]]
>
>>[!info] Osservazione
>>Le stringhe da stampare sono $2^n$ e stampare una stringa lunga $n$ costa $\Theta(n)$
>
>>[!done]-
>>Il meglio che ci si può augurare per un algoritmo che risolve questo problema è una complessità $\Omega(2^n\cdot n)$
>>
>>Implementazione:
>>```python
>>def es(n, sol=[]):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	sol.append(0)
>>	es(n, sol)
>>	sol.pop()
>>	sol.append(1)
>>	es(n, sol)
>>	sol.pop()
>>```
>>
>>L’albero binario di ricorsione ha $2\cdot2^n$ nodi di cui $2^n$ foglie:
>>- ciascun nodo interno richiede tempo $O(1)$
>>- ciascuna foglia richiede tempo $\Theta(n)$ (il print costa $\Theta(n)$)
>>
>>La complessità dell’algoritmo è $\Theta(2^n)+2^n\Theta(n)=\Theta(2^n\cdot n)$

>[!question] Progettare un algoritmo che prendere come parametri due interi $n$ e $k$, con $0\leq k\leq n$, e stampa tutte le stringhe binarie lunghe $n$ che contengono al più $k$ uni
>Ad esempio per $n=4$ e $k=2$ delle $2^4=16$ stringhe lunghe $n$ bisogna stampare le seguenti $11$:
>```
>0000, 0001, 0010, 0100, 1000, 0011, 0101, 1001, 0110, 1010, 1100
>```
>
>>[!info] Osservazione
>>- $k=0$ → $1$ stringa
>>- $k=1$ → $n+1$ stringhe (uno in qualsiasi posizione e tutti zeri)
>
>>[!done]-
>>Un possibile algoritmo che risolve il problema in $\Omega(2^n\cdot n)$:
>>```python
>>def es(n, k, sol=[]):
>>	if len(sol)==n and sol.count(1)<=k:
>>		print(sol)
>>		return
>>	sol.append(0)
>>	es(n, k, sol)
>>	sol.pop()
>>	sol.append(1)
>>	es(n, k, sol)
>>	sol.pop()
>>```
>>
>>Indichiamo con $S(n,k)$ il numero di stringhe che bisogna stampare. Un buon algoritmo per questo problema dovrebbe avere una complessità proporzionale alle stringhe da stampare, vale a dire $O(S(n,k)\cdot n)$
>>
>>>[!example]
>>>Ad esempio per $k=2$ si ha:
>>>$$S(n,k)=1+n+\binom{n}{2}=\Theta(n^2)$$
>>>e quindi un buon algoritmo per $k=2$ dovrebbe avere complessità polinomiale $O(n^3)$ mentre l’algoritmo proposto ha complessità esponensiale $\Theta(2^n\cdot n)$ (indipendente da $k$)
>>
>>![[Pasted image 20250515003611.png]]
>>>[!info] Osservazione
>>>Inutile generare nell’albero di ricorsione nodi che non hanno possibilità di portare a soluzioni da stampare
>>
>>![[Pasted image 20250515003712.png]]
>>
>>Implementazione:
>>```python
>>def es(n, k, sol=[], uni=0):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	sol.append(0)
>>	es(n, k, sol, uni)
>>	sol.pop()
>>	if uni<k:
>>		sol.append(1)
>>		es(n, k, sol, uni+1)
>>		sol.pop()
>>```
>>
>>Si consideri un algoritmo di enumerazione basato sul backtracking dove l’albero di ricorsione ha altezza $h$, il costo di una foglia è $g(n)$ e il costo di un nodo interno è $O(f(n))$
>>Se l’algoritmo gode della seguente proprietà:
>>$$\text{un nodo viene generato solo se ha la possibilità di portare ad una foglia da stampare}$$
>>
>>Allora la complessità dell’algoritmo è proporzionale al numero di cose da stampare $S(n)$, più precisamente la complessità dell’algoritmo è:
>>$$O(S(n)\cdot h\cdot f(n)+S(n)\cdot g(n))$$
>>questo perché:
>>- il costo totale dei nodi foglia sarà $O(S(n)\cdot g(n))$ (in quanto solo le foglie da enumerare verranno generate)
>>- i nodi interni dell’albero che verranno effettivamente generati saranno $O(S(n)\cdot h)$ (in quanto ogni nodo interno generato apparterrà ad un cammino che parte dalla radice e arriva ad una delle $S(n)$ foglie da enumerare)
>>
>>Analizzando l’algoritmo scritto la proprietà di generare un nodo solo se questo può portare ad una delle $S(n,k)$ foglie da stampare è rispettata. Inoltre $h=n$, $g(n)=\Theta(n)$, $f(n)=O(1)$
>>Quindi la complessità è:
>>$$S(n,k)\cdot n\cdot O(1)+S(n,k)\cdot \Theta(n)=\Theta(S(n,k)\cdot n)$$
>>e l’algoritmo risulta ottimale
>>
>>La complessità dell’algoritmo è $O(n^{k+1})$ infatti: $S(n,k)=\binom{n}{0}+\binom{n}{0}+\dots+\binom{n}{k}<2\cdot n^k$

>[!question] Progettare un algoritmo che prende come parametro $n$ e stampa le stringhe che non hanno $3$ uni consecutivi
>>[!done]-
>>Implementazione:
>>```python
>>def es(n, sol=[]):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	sol.append(0)
>>	es(n, k, sol)
>>	sol.pop()
>>	if len(sol)<2 or sol[-1]!=1 or sol[-2]!=1:
>>		sol.append(1)
>>		es(n, sol)
>>		sol.pop()
>>```
>>
>>La complessità è:
>>$$O(S(n)\cdot h\cdot f(n)+S(n)\cdot g(n))=O(S(n)\cdot n\cdot \Theta(1)+S(n)\cdot n)=O(S(n)\cdot n)$$



```python
>>def es(n, sol=[]):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	sol.append(0)
>>	es(n, k, sol)
>>	sol.pop()
>>	if len(sol)<2 or sol[-1]!=1 or sol[-2]!=1:
>>		sol.append(1)
>>		es(n, sol)
>>		sol.pop()
>>```
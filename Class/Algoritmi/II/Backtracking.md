---
Created: 2025-05-15
Class: "[[Algoritmi]]"
Related:
---
---
## Index
- [[#Esercizi|Esercizi]]
---
## Esercizi liste

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

>[!question] Progettare un algoritmo che prendere come parametri due interi $n$ e $k$, con $0\leq k\leq n$, e stampa tutte le stringhe binarie lunghe $n$ che contengono esattamente $k$ uni
>Ad esempio per $n=6$ e $k=3$ delle $2^6=64$ stringhe lunghe $n$ bisogna stampare le seguenti $20$:
>```
>000111, 0011011, 001101, 001110, 010011, 010101, 010110, 011001, 011010, 011100, 100011, 100101, 100110, 101001, 101010, 101100, 110001, 110010, 110100, 111000
>```
>
>>[!done]-
>>Rispetto all’esercizio precedente aggiungiamo ora una funzione di taglio anche nel caso in cui al prefisso viene aggiunto uno zero. Bisogna assicurarsi infatti che sia sempre completare quel prefisso in modo da ottenere una stringa valida da stampare
>>
>>Implementazione
>>```python
>>def es(n, k, sol=[], uni):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	if n-len(sol)>k-uni:
>>		sol.append(0)
>>		es(n, k, sol)
>>		sol.pop()
>>	if uni<k:
>>		sol.append(1)
>>		es(n, k, sol, uni+1)
>>		sol.pop()
>>```
>>
>>Analizzando l’algoritmo scritto la proprietà di generare un nodo solo se questo può portare ad una delle $S(n,k)$ foglie da stampare è rispettata. Inoltre $h=n$, $g(n)=\Theta(n)$, $f(n)=O(1)$
>>Quindi la complessità è:
>>$$S(n,k)\cdot n\cdot O(1)+S(n,k)\cdot \Theta(n)=\Theta(S(n,k)\cdot n)$$
>>e l’algoritmo risulta ottimale
>>
>>La complessità dell’algoritmo è $O(n^{k+1})$ infatti: $S(n,k)=\binom{n}{k}<n^k$

---
## Esercizi matrici

>[!question] Progettare un algoritmo che prende come parametro un intero $n$ e stampa tutte le matrici binarie $n\times n$
>Ad esempio per $n=2$ bisogna stampare le seguenti $2^4=16$ matrici:
>![[Pasted image 20250515233047.png]]
>
>>[!done]-
>>Implementazione:
>>```python
>>def es(n):
>>	sol=[[0]*n for _ in range(n)]
>>	es1(n, sol)
>>
>>def es1(n, sol, i=0, j=0):
>>	if i==n:          # non necessario j, infatti i==n vuol dire che ho
>>		for k in sol: # superato la fine
>>			print(k)
>>		print()
>>	i1,j1 = i,j+1
>>	if j1==n:
>>		i1,j1 = i+1,0
>>	sol[i][j]=0
>>	es1(n, sol, i1, j1)
>>	sol[i][j]=1
>>	es1(n, sol, i1, j1)
>>```
>>
>>L’albero di ricorsione è binario e di altezza $n^2$; ha dunque $2^{n^2}-1$ nodi interni e $2^{n^2}$ foglie. Ciascun nodo interno richiede tempo $O(1)$ e ciascuna foglia $\Theta(n^2)$
>>L’algoritmo ha complessità $O(2^{n^2}n^2)$
>>
>>Poiché le matrici da stampare sono $2^{n^2}$ e la stampa di una matrice richiede $\Theta(n^2)$, qualunque algoritmo di per questo problema richiede $\Omega(2^{n^2}n^2)$
>>
>>L’algoritmo è ottimo

>[!question] Progettare un algoritmo che prende come parametro un intero $n$ e stampa tutte le matrici binarie $n\times n$ in cui righe e colonne risultano ordinate in modo non decrescente
>Ad esempio per $n=2$ bisogna stampare le seguenti $2^4=16$ matrici quadrate $3\times 3$ bisogna stampare le seguenti $6$:
>![[Pasted image 20250515235045.png]]
>
>La complessità dell’algoritmo deve essere $O(n^2S(n))$ dove $S(n)$ è il numero di matrici da stampare
>
>>[!done]-
>>Implementazione:
>>```python
>>def es(n):
>>	sol=[[0]*n for _ in range(n)]
>>	es1(n, sol)
>>
>>def es1(n, sol, i=0, j=0):
>>    if i==n:          # non necessario j, infatti i==n vuol dire che ho
>>        for k in sol: # superato la fine
>>            print(k)
>>        print()
>>        return
>>    i1,j1 = i,j+1
>>    if j1==n:
>>        i1,j1 = i+1,0
>>    if (i==0 or sol[i-1][j]==0) and (j==0 or sol[i][j-1]==0):
>>        sol[i][j]=0
>>        es1(n, sol, i1, j1)
>>    sol[i][j]=1
>>```
>>
>>L’albero di ricorsione è binario e di altezza $n^2$ e solo i nodi che portano ad una delle $S(n)$ soluzioni vengono effettivamente generati
>>
>>I nodi interni all’albero di ricorsione effettivamente generati saranno $O(S(n)\cdot n^2)$ e la foglie effettivamente generate saranno $S(n)$. Ciascun nodo interno richiede tempo $O(1)$ e ciascuna foglia richiede $O(n^2)$
>>
>>Il tempo totale sarà $O(S(n)\cdot n^2)+O(S(n)\cdot n^2)=O(S(n)\cdot n^2)$
>>L’algoritmo proposto è ottimo

>[!question] Progettare un algoritmo che prende come parametro un intero $n$ e stampa tutte le permutazioni dei numeri da $0$ a $n-1$
>Ad esempio per $n=4$ bisogna stampare le seguenti $4!=24$ permutazioni:
>![[Pasted image 20250516002328.png]]
>
>>[!info] Albero delle permutazioni
>>![[Pasted image 20250516002529.png]]
>>
>>L’albero delle permutazioni ha $\Theta(n!)$ foglie e $\Theta(n!)$ nodi interni
>>$$\text{nodi int.}=\sum^n_{i=0} \frac{n!}{i!}<n!\cdot \sum^{\infty}_{i=0} \frac{1}{i!}\leq n!\cdot \sum^{\infty}_{i=0} \frac{2}{2^i}=n! \frac{2}{1-1/2}=4\cdot n!$$
>
>>[!done]-
>>Implementazione:
>>```python
>>def es(n):
>>	preso=[0]*n
>>	es1(n, [], preso)
>>
>>def es1(n, sol, preso):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	for j in range(n):
>>	    if preso[j]==0:
>>		    sol.append(k)
>>		    preso[j]=1
>>		    es1(n, sol, preso)
>>		    sol.pop()
>>		    preso[j]=0
>>```
>>
>>L’albero delle permutazioni ha $\Theta(n!)$ nodi interni e $n!$ foglie e ciascun nodo interno richiede tempo $\Theta(n)$ e ciascuna foglia richiede $\Theta(n)$
>>
>>Il tempo totale sarà $\Theta(n!\cdot n)$
>>L’algoritmo proposto è ottimo

>[!question] Progettare un algoritmo che prende come parametro un intero $n$ e stampa tutte le permutazioni dei numeri da $0$ a $n-1$ dove nelle posizioni pari compaiono numeri pari e viceversa
>Ad esempio per $n=5$ delle $5!=125$ permutazioni bisogna stampare le seguenti $12$:
>![[Pasted image 20250516004444.png]]
>
>>[!done]-
>>Implementazione:
>>```python
>>def es(n):
>>	preso=[0]*n
>>	es1(n, [], preso)
>>
>>def es1(n, sol, preso):
>>	if len(sol)==n:
>>		print(sol)
>>		return
>>	for j in range(n):
>>	    if preso[j]==0 and j%2==len(sol)%s:
>>		    sol.append(k)
>>		    preso[j]=1
>>		    es1(n, sol, preso)
>>		    sol.pop()
>>		    preso[j]=0
>>```
>>
>>L’albero delle permutazioni è di altezza $n$ e solo i nodi che portano ad una delle $S(n)$ soluzioni vengono effettivamente generati
>>
>>I nodi effettivamente generati saranno $O(S(n)\cdot n)$ e le foglie effettivamente generate saranno $S(n)$. Ciascun nodo interno richiede tempo $\Theta(n)$ e ciascuna foglia richiede tempo $\Theta(n)$
>>
>>Il tempo totale sarà $O(S(n)\cdot n)\cdot \Theta(n)+S(n)\cdot \Theta(n)=O(S(n)\cdot n^2)$
>>L’algoritmo proposto è ottimo

p
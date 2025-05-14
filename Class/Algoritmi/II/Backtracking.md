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
>>[!done]
>>Un possibile algoritmo che risolve il problema in $\Omega(2^n\cdot n)$
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



>>[!NOTE]
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
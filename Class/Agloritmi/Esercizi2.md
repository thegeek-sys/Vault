---
Created: 2025-03-03
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## 28/02
### #1
In un grafo diretto un pozzo è un nodo senza archi uscenti. In un grafo diretto un pozzo universale è un pozzo verso cui tutti gli altri nodi hanno un arco
![[Pasted image 20250303115954.png]]

>[!info]
>- In un grafo diretto possono essere presenti fino a $n$ pozzi
>- Il pozzo universale se c’è è unico

![[Pasted image 20250303120210.png|center|300]]

Trovare un algoritmo di tempo $O(n^2)$ per verificare se un grafo diretto $M$ ha un pozzo universale
```python
def test_pozzoU(x,M):
	for j in range(len(M)):
		if M[x][j]: return False
	for i in range(len(M)):
		if i!=x and M[i][x]==0: return False
	return True

def pozzoU2(M):
	for x in range(len(M)):
		if test_pozzoU(x,M):
			return True
	return False
```

Trovare un algoritmo di tempo $O(n)$ per verificare se un grafo diretto $M$ ha un pozzo universale
Possiamo dire innanzitutto che se:
$$
M[i][j] = \begin{cases} 1 & \text{se } i \text{ non è pozzo (universale)} \\ 0 & \text{se } j \text{ non è pozzo (universale)} \end{cases}
$$
Infatti se $M[i][j]=1$ vuol dire che ci sta un arco $i\rightarrow j$ quindi $i$ non è pozzo universale, mentre se $M[i][j]=0$ vuol dire che non ci sta nessun arco $i\rightarrow j$ quindi non entra un arco in $j$ (non è pozzo universale)

Quindi con un semplice test posso eliminare uno dei nodi dai possibili pozzi universali. Dopo $n-1$ test mi resta un unico nodo da controllare

```python
def pozzoU2(M):
	L=[x for x in range(len(M))] # un elemento per nodo
	while len(L)>1:
		# rimuovo due nodi alla volta e ne reinserisco uno ogni volta
		# finché non mi ritrovo con un solo nodo (possibile pozzo 
		# universale) seguendo le regole dette sopra
		a=L.pop()
		b=L.pop()
		if M[a][b]:
			L.append(b)
		else:
			L.append(a)
	x=L.pop()
	# testo se l'ultimo nodo è pozzo universale
	for j in range(len(M)):
		if M[x][j]: return False
	for i in range(len(M)):
		if i!=x and M[i][x]==0: return False
	return True
```

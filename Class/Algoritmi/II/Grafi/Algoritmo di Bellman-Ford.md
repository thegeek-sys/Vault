---
Created: 2025-03-28
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction

>[!question] Problema
>Dato un grafo diretto e pesato $G$ in cui i pesi degli archi possono essere anche negativi e fissato un suo nodo $s$, vogliamo determinare il costo minimo dei cammini che conducono da $s$ a tutti gli altri nodi del grafo. Se non esiste un cammino verso un determinato nodo il costo sarà infinito

In questo primo problema però non è considerata la possibilità di poter avere un **ciclo negativo**, ovvero un ciclo diretto in un grafo in cui la somma dei pesi degli archi che lo compongono è negativa

>[!example]
>![[Pasted image 20250328011833.png|350]]
>Il ciclo evidenziato in figura è negativo di costo $5-3-6+4-2=\textcolor{red}{-2}$

Infatti se in un cammino tra i nodi $s$ e $t$ è presente un nodo che appartiene ad un ciclo negativo allora non esiste un cammino minimo tra $s$ e $t$ (è $-\infty$)
![[Pasted image 20250328012200.png|400]]

Se per il ciclo $W$ si ha costo $\text{costo}(W)<0$, ripassando più volte attraverso il ciclo $W$ possiamo arbitrariamente abbassare il costo del cammino da $s$ a $t$

Alla luce di quanto appena detto sui cicli negativi, la formulazione del problema non era del tutto corretta. Ecco di seguito la versione corretta

>[!question] Problema
>Dato un grafo diretto e pesato $G$ in cui i pesi degli archi possono essere anche negativi **ma che non contiene cicli negativi**, e fissato un suo nodo $s$, vogliamo determinare il costo minimo dei cammini che conducono da $s$ a tutti gli altri nodi del grafo. Se non esiste un cammino verso un determinato nodo il costo sarà infinito

---
## Algoritmo di Bellman-Ford
Per risolvere questo problema, come abbiamo già visto, non è possibile usare l’algoritmo di Dijsktra. Useremo quindi l’**algoritmo di Bellman-Ford**, di complessità $O(n^2+m\cdot n)$

>[!info] Proprietà
>Se il grafo $G$ non contiene cicli negativi, allora per ogni nodo $t$ raggiungibile dalla sorgente $s$ esiste un cammino di costo minimo che attraversa al più $n-1$ archi

Infatti, se un cammino avesse più di $n-1$ archi, allora almeno un nodo verrebbe ripetuto, formando un ciclo. Poiché il grafo non ha cicli negativi, rimuovere eventuali cicli dal cammino **non aumenta** il suo costo complessivo, di conseguenza esiste sempre un cammino ottimale di lunghezza $n-1$
Questo garantisce che il costo minimo può essere calcolato considerando solo cammini di questa lunghezza, è quindi possibile considerare sottoproblemi che si ottengono limitando la lunghezza dei cammini.
![[Pasted image 20250328013014.png|600]]

Definiamo così la seguente tabella di dimensione $n\times n$
$$
T[i][j]=\text{costo di un cammino minimo da }s\text{ al nodo }j\text{ attraversando al più }i\text{ archi}
$$
Calcoleremo la soluzione al nostro problema determinando i valori della tabella. Infatti il costo minimo per andare da $s$ (sorgente) al generico nodo $t$ sarà $T[n-1][t]$

>[!example]
>Alla creazione, la tabella sarà del tipo:
>![[Pasted image 20250331105556.png]]
>
>Se $n-1$ e $n-2$ sono uguali allora ho il costo minimo, se sono diversi vuol dire che ci sta un ciclo negativo (i costi sono calcolati in base alla riga precedente)

I valori della prima riga della tabella $T$ sono ovviamente tutti $+\infty$ tranne $T[0][s]$ che vale $0$. Inoltre $T[i][s]=0$ per ogni $i>0$

Resta da definire la regola che permette di calcolare i valori delle celle $T[i][j]$ con $j\neq s$ della riga $i>0$ in funzione delle celle già calcolare della riga $i-1$
Distinguiamo due casi a seconda che il cammino di lunghezza al più $i$ da $s$ a $j$ abbia lunghezza esattamente $i$ o inferiore a $i$:
- nel primo caso ovviamente si ha $T[i][j]=T[i][j-1]$
- nel secondo caso deve invece esistere un cammino minimo di lunghezza al più $i-1$ ad un nodo $x$ e poi un arco che da $x$ mi porta a $j$, ovvero $T[i][j]=\underset{(x,j)\in E}{\text{min}}\left(T[i-1][x]+\text{costo}(x,j)\right)$. Tradotta questo formula significa “tra tutti gli archi $x$ che portano a $j$, scegli quello che costa meno prendendolo dalla riga precedente e sommando il costo di $(x,j)$”

Non sapendo in quale dei due casi siamo la formula giusta è:
$$
T[i][j]=\underset{(x,j)\in E}{\text{min}}\Bigl(T[i-1][j],\;\;T[i-1][x]+\text{costo}(x,j)\Bigl)
$$

Riassumendo le celle della tabella possono essere riempite per righe in base a questa regola:
$$
T[i][j]=
\begin{cases}
0&\text{se }j=s \\
+\infty&\text{se }i=0 \\
\underset{(x,j)\in E}{\text{min}}\Bigl(T[i-1][j],\;\;T[i-1][x]+\text{costo}(x,j)\Bigl)&\text{altriementi}
\end{cases}
$$

>[!hint]
>Per un’implementazione efficiente, poiché nel calcolo della formula è necessario più volte conoscere gli archi entranti del generico nodo $j$, conviene precalcolare il grafo trasposto $GT$ di $G$. In questo modo, in $GT[j]$ avremo l’elenco di tutti i nodi $x$ tali che in $G$ esiste un arco da $x$ a $j$.
>
>Questo permette di accedere rapidamente agli archi entranti di un nodo, migliorando l’efficienza

### Implementazione

```python
def trasposto(G):
	GT = [[] for _ in G]
	for i in range(len(G)):
		for j,costo in G[i]:
			GT[j].append((i,costo))
	return GT

def costo_cammini(G, s):
	T = [[float('inf')]*len(G) for _ in range(len(G))]
	T[0][s] = 0
	GT = trasposto(G)
	for i in range(1,n)
		for j in range(n):
			T[i][j] = T[i-1][j]
			if j!=s:
				for x,costo in GT[j]:
					T[i][j]=min(T[i][j], T[i-1][x]+costo)
	return T[len(G)-1]

# >> G = [
# [(1,3),(3,6)],
# [],
# [(1,1)],
# [(2,-5)],
# [(0,2)]
# ]
# >> costo_cammini(G,0)
# [0,2,1,6,inf]
# >> costo_cammini(G,4)
# [-2,0,-1,4,0]
```
La complessità è:
- l’inizializzazione della tabella $T$ costa $\Theta(n^2)$
- la costruzione del grafo trasposto $GT$ richiede tempo $O(n+m)$
- per i tre $\verb|for|$ annidati è ovvio il limite superiore $O(n^3)$, ma facciamo un’analisi più attenta. I due $\verb|for|$ più interni hanno costo totale $\Theta(m)$, infatti il tempo richiesto è sostanzialmente quello di scorrere tutte le lista di adiacenza del grafo $GT$ che hanno lunghezza totale $m$. Se ad esempio nel primo $\verb|for|$ interno si itera su tutti i nodi (caso peggiore), nel secondo non si entrerà neanche una volta (non ci saranno archi entranti)

La complessità complessiva è $O(n^2+mn)$

---
## Trovare anche i cammini
Per ritrovare anche i cammini (oltre al loro costo) con la tabella $T$ bisogna calcolare anche l’albero $P$ dei padri (cammini minimi). Questo si può fare facilmente mantenendo per ogni nodo $j$ il suo predecessore, cioè il nodo $u$ che precede $j$ nel cammino. Il valore di $P[j]$ andrà aggiornando ogni volta che il valore di $T[k][j]$ cambia (ovvero diminuisce) in quanto abbiamo trovato un cammino migliore

```python
def costo_cammini1(G, s):
	T = [[float('inf')]*len(G) for _ in range(len(G))]
	P = [-1]*len(G)
	GT = trasposto(G)
	
	T[0][s] = 0
	P[s] = s
	for i in range(1,n)
		for j in range(n):
			if j==s:
				T[k][j] = 0
			else:
				for x,costo in GT[j]:
					if T[k-1][x]+costo < T[k][j]:
						T[k][j] = T[k-1][x]+costo
						P[j] = x
	return T[len(G)-1], P
```
Con questa implementazione al temine dell’algoritmo:
- $T[n-1][j]\neq+\infty$ indica che $j$ è raggiungibile da $s$ (in questo caso $P[j]$ conterrà il nodo che precede $j$ nel cammino minimo da $s$ a $j$)
- $T[n-1][j]=+\infty$ indica che $j$ non è raggiungibile a partire da $s$ (in questo caso $P[j]$ conterrà il valore $-1$)

---
## Ottimizzazioni

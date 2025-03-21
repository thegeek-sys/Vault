---
Created: 
Class: 
Related: 
Completed:
---
---
## Introduction
Consideriamo un insieme di computer (server e/o router) che devono essere connessi tramite cavi a formare una rete in modo che ogni computer possa comunicare con ogni altro o tramite un cavo che li collega direttamente o passando per gli altri computer. Ogni possibile collegamento tramite cavo ha un costo. Un esempio è mostrato nella figura seguente:
![[Pasted image 20250321101549.png|450]]

Quindi vogliamo installare alcuni dei collegamenti possibili in modo tale da garantire la connessione della rete e al contempo minimizzare il costo totale

Una possibile soluzione (di costo 22):
![[Pasted image 20250321101724.png|450]]

>[!info]
>Il costo dello spanning tree è la somma dei costi dei singoli archi

Il problema può essere rappresentato tramite un grafo pesato e connesso $G$ i cui nodi sono i computer, gli archi sono i possibili collegamenti con i loro costi

![[Pasted image 20250321102251.png|400]]
![[Pasted image 20250321102329.png|400]]

>[!warning]
>Nel grafo soluzione (con gli archi in rosso in figura) non sono mai presenti cicli.
>Infatti l’eliminazione di qualunque arco del ciclo non farebbe perdere la connessione e diminuirebbe il costo della soluzione

Il sosttoinsieme degli archi del grafo che formano la soluzione è dunue un albero (grafo connesso aciclico). Andiamo quindi alla ricerca in $G$ di un albero che “copre” l’intero grafo e la somma dei costi dei suoi archi sia minima. Questo problema prende il nome di **minimo albero di copertura** (*minimum spanning tree*)

---
## Algoritmo di Kursal
Per risolvere il problema del minimo albero di copertura, data la sua importanza esistono diversi algoritmi per risolvere questo problema. Ora analizzeremo **l’algoritmo di Kursal**:
- Parti con il grafo $T$ che contiene tutti i nodi di $G$ e nessun arco di $G$
- Considera uno dopo l’altro gli archi del grafo $G$ in ordine di costo creascente
- Se l’arco forma un ciclo in $T$ con archi già presi allora non prenderlo altrimenti inseriscilo in $T$
- Al termine restituisci $T$

>[!hint]
>L’algoritmo rientra perfettamente nel paradigma della **tecnica greedy**:
>- la sequenza di decisioni irrevocabili → decidi per ciascun arco di $G$ se inserirlo o meno in $T$. Una volta deciso cosa fare dell’arco non ritornare più su questa decisione
>- le decisioni vengono prese in base ad un criterio “locale” → se l’arco crea un ciclo non lo prendi, in caso contrario lo prendi in quanto è il meno costoso a non creare cicli tra gli archi che restano da considerare

>[!example]
>![[Pasted image 20250321104501.png|350]]
>![[Pasted image 20250321104543.png|450]]
>![[Pasted image 20250321104600.png|450]]

```
kursal(G):
	T=set()
	inizializza E con gli archi di G
	while E!=[]
		estrai da E un arco (x,y) di peso minimo
		if l'inserimento di (x,y) in T non crea ciclo con gli archi in T:
			inserisci arco (x,y) in T
	return T
```

>[!done] Dimostrazione correttezza
>Dobbiamo far vedere che al termine dell’algoritmo, $T$ è albero di copertura e che non c’è un altro albero che costa meno
>
>Lo dimostreremo **per assurdo**
>
>##### Produce un albero di copertura
>Supponiamo che al termine in $T$ ci sia più di una componente ($T$ non è connesso). 
>Se è così vuol dire che sono presenti almeno 2 componenti nel grafo $T$ e il che vuol dire che l’arco che connetteva le due componenti (presente in $G$) non è stato scelto.
>
>Ma l’unico motivo per cui un arco non viene scelto è poiché questo crea un ciclo, ma un arco che connette due componenti non crea mai un ciclo
>**CONTRADDIZIONE**
>
>##### Non c’è un albero di copertura per $G$ che costa meno di $T$
>Tra tutti gli alberi di copertura di costo minimo per $G$ prendiamo quello che differisce nel minor numero di archi da $T$; sia questo grafo $T^*$.
>
>Supponiamo per assurdo che $T$ differisca da $T^*$. Faremo vedere che questa assunzione porterebbe all’assurdo perché avrebbe come conseguenza l’esistenza di un altro albero di copertura di costo minimo per $G$ che differisce da $T$ in meno archi di $T^*$
>
>Considera l’ordine $e_{1},e_{2},\dots$ con cui gli archi sono presi in considerazione nel corso dell’algoritmo e sia $e$ il primo arco preso che non compare in $T^*$. Se inserisco $e$ in $T^*$ si forma un ciclo $C$. Il ciclo $C$ contiene almeno un arco $e'$ che non compare in $T$ (infatti non tutti gli archi del ciclo $C$ sono in $T$ altrimenti $e$ non sarebbe stato preso dall’algoritmo di Kursal).
>
>Considera ora l’albero $T'$ che ottengo da $T^*$ inserendo l’arco $e$ ed eliminando l’arco $e'$. Il costo del nuovo albero $T'$ (che è $\text{costo}(T^*)-\text{costo}(e')+\text{costo}(e)$) non può aumentare rispetto a quello di $T^*$; infatti $\text{costo}(e)\leq \text{costo}(e')$ poiché tra i due archi $e$ ed $e'$ che non creavano ciclo, Kursal ha considerato prima l’arco $e$, ma allora $T'$ è un altro albero di copertura ottimo che differisce da $T$ in meno archi di quanto faccia $T^*$, il che contraddice l’ipotesi che $T^*$ differisce da $T$ nel minor numero di archi
>**CONTRADDIZIONE**
>
>![[correttezza.png]]

### Implementazione
**Idee**:
- con un pre-processing ordino gli archi in $E$ in ordine crescente cosicché l’estrazione da $E$ dell’arco di costo minimo costi $O(1)$
- verifico che l’arco $(x,y)$ non formi ciclo in $T$ controllando se $y$ è raggiungibile da $x$ in $T$

![[Pasted image 20250321113013.png]]

```python
def connessi(x,y,T):
	def connessiR(a,b,T):
		visitati[a] = 1
		for z in T[a]:
			if z == b: return True
			if not visitati[z] and connessiR(z,b,T): return True
		return False
	visitati = [0]*len(T)
	return connessiR(x,y,T)

def kursal(G):
	E = [(c,u,v) for u in range(len(G)) for v,c in G[u] if u<v]
	E.sort(reverse=True)
	T = [[] for _ in G]
	
	while E:
		c,x,y = E.pop()
		if not connessi(x,y,T):
			T[x].append(y)
			T[y].append(x)
	return T

# >> G = [
#	[(1,25),(2,10),(3,35)],
#	[(0,25),(2,20),(3,42)],
#	[(0,10),(1,20),(3,40)],
#	[(0,35),(1,42),(2,40)]
#	]
# >> kursal(G)
# [[2,3], [2], [0,1], [0]]
```
L’ordinamento esterno al $while$ ci costa $O(m\log m)=O(m\log n)$. Il $while$ vine iterato $m$ volte (ad ogni passo viene inserito un arco in $T$)
L’estrazione dell’arco $(a,b)$ di costo minimo da $E$ richiede tempo $O(1)$ e controllare che l’arco $(a,b)$ non crei un ciclo in $T$ con la procedura $\verb|connessi(a,b,T)|$ richiede il costo di una visita in un grafo aciclico quindi $O(n)$

La complessità di questa implementazione è $O(m\cdot n)$

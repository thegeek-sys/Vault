---
Created: 2025-03-26
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Introduction
**Union-Find**, noto anche come *Disjoint Set Union* (DSU), è una struttura dati per gestire insiemi disgiunti. E’ utilizzato per operazioni di unione e ricerca efficienti su insiemi disgiunti

Le tre operazioni fondamentali di Union-Find sono:
1. $Crea(S)$ → restituisce una struttura dati Union-Find sull’insieme $S$ di elementi dove ciascun elemento è in un insieme separato
2. $Find(x,C)$ → restituisce il nome dell’insieme della struttura dati $C$ a cui appartiene l’elemento $x$ (a quele componente appartiene $x$)
3. $Union(A,B,C)$ → modifica la struttura dati $C$ fondendo la componente $A$ con la componente $B$ e restituisce il nome della nuova componente

Una gestione efficiente di insiemi disgiunti è utile in diversi contesti tra cui l’evoluzione di un grafo nel tempo attraverso l’aggiunta di archi. In questo caso gli insiemi disgiunti rappresentano le componenti connesse del grafo

Se viene aggiunto l’arco $(u,v)$, al grafo, si verifica innanzitutto se $u$ e $v$ sono nella stessa componente connessa. Se $A=find(u)$ e $B=find(v)$ risultano distinti allora l’operazione $Union(A,B)$ può essere usata per unire le due componenti

### Nome dell’insieme
Discutiamo brevemente su cosa intendiamo con **nome dell’insieme** (ad esempio quello ritornato dalla funzione $find$ su un elemento $x$). C’è un’ampia flessibilità nella scelta, come risulta dalle varie implementazioni la cosa importante è che $find(u)=find(v)$ se e solo se $u$ e $v$ sono nello stesso insieme.
La scelta fatta nelle implementazioni che seguono. è quella di scegliere come **nome dell’insieme quello di un particolare elemento dell’insieme stesso**
Come primo approccio possiamo pensare di assegnare all’insieme il nome dell’elemento massimo in esso contenuto

---
## Prima implementazione
Probabilmente il modo più semplice di implementare questa struttura dati per $n$ elementi è di mantenere il vettore $C$ delle $n$ componenti.
All’inizio ogni elemento è in un insieme distinto vale a dire $C[i] =i$. Quando la componente $i$ viene fusa con la componente $j$, se $i>j$ allora tutte le occorrenze di $j$ nel vettore $C$ verranno sostituite da $i$ (se $i<j$ accadrà il contrario)

```python
def Crea(G):
	C = [i for i in range(len(G))]
	return C

def Find(u,C):
	return C[u]

def Union(a,b,C):
	if a>b:
		for i in range(len(C))
			if C[i] == b:
				C[i] = a
	else:
		for i in range(len(C)):
			if C[i] == a:
				C[i] = b
```
Costo computazionale:
- $Crea()$ → costo $\Theta(n)$
- $Find()$  → costo $\Theta(1)$
- $Union()$ → costo $\Theta(n)$

### Miglioramento dei costi computazionali di $\verb|UNION|$
Meglio però bilanciare i costi: rendere meno costosa la $\verb|UNION|$ anche a costo di pagare qualcosa in più per la $\verb|FIND|$

Uso il vettore dei padri:
- $\verb|FIND|$ → quando voglio sapere in che componente si trova un nudo devo semplicemente risalire alla sua radice. $\Theta(n)$
- $\verb|UNION|$ → quando fondo due componenti una diventa figlia dell’altra. $O(1)$

![[Pasted image 20250326110049.png|500]]
![[Pasted image 20250326110105.png|500]]

```python
def Crea(G):
	C = [i for i in range(len(G))]
	return C

def Find(u,C):
	while u != C[u]:
		u = C[u]
	return u

def Union(a,b,C):
	if a>b:
		C[b]=a
	else:
		C[a]=b
```
Costo computazionale:
- $Crea()$ → costo $\Theta(n)$
- $Find()$  → costo $\Theta(n)$
- $Union()$ → costo $\Theta(1)$

---
## Bilanciamento dei costi computazionali
Non voglio però che l’operazione $\verb|FIND|$ abbia un costo elevato quindi è importante che i cammini per raggiungere le radici del vettore dei padri non diventino troppo lunghi.
Per evitare questo problema, è preferibile mantenere gli alberi bilanciati

Quando esegui la $\verb|UNION|$ per fondere due componenti scelgo sempre come nuova radice la componente che contiene il maggior numero di elementi

L’intuizione è che in questo modo per almeno la metà dei nodi presenti nelle due componenti coinvolte nella fusione la lunghezza del cammino non aumenta.
Fondendo le componenti con questo accorgimento garantiamo le seguenti proprietà: se un insieme ha altezza $h$ allora l’insieme contiene almeno $2^h$ elementi

Dalla proprietà deduciamo che l’altezza delle componenti non potrà mai superare $\log_{2}n$ (poiché in caso contrario avrei nella componente più di $n$ nodi il che è assurdo)

### Implementazione
In questa implementazione della Union-Find devo fare in modo che ai nodi radice sia associato anche il numero di elementi che la componente contiene.
Ogni elemento è caratterizzato da una coppia $(x,\text{numero})$ dove $x$
è il nome dell’elemento e $\text{numero}$ è il numero di nodi nell’albero radicato in $x$

```python
def Crea(G):
	C = [(i,1) for i in range(len(G))]
	return C

def Find(u, C):
	while u != C[u]:
		u = C[u]
	return u

def Union(a, b, C):
	tota, totb = C[a][1], C[b][1]
	if tota >= totb:
		C[a] = (a, tota + totb)
		C[b] = (a, totb)
	else:
		C[b] = (b, tota + totb)
		C[b] = (a, totb)
```
Costo computazionale:
- $Crea()$ → costo $\Theta(n)$
- $Find()$  → costo $\Theta(\log n)$
- $Union()$ → costo $\Theta(1)$

>[!info] Proprietà
>Se una componente ha altezza $h$ allora la componente contiene almeno  $2^h$ nodi
>
>>[!done] Dimostrazione
>>Assumiamo per assurdo durante una delle fusioni si sia formata una nuova componente di altezza $h$ che non rispetta la proprietà.
>>
>>Considera la prima volta ciò che accade e siano $ca$ e $cb$ le componenti che si fondono.
>>Possono accadere due cose:
>>- **$ca$ e $cb$ erano componenti della stessa altezza**, allora avevano entrambe altezza $h-1$ ed ognuna aveva almeno $2^{h-1}$ elementi (perché  nelle fusioni precedenti la proprietà era smepre verificata). Quindi il numero totale di elementi della nuova componente è $2^{h-1}+2^{h-1}=2^h$ e la proprietà è verificata
>>- **$ca$ e $cb$ avevano diverse altezze**, allora l’altezza dopo la fusione è quella della componente di altezza maggiore che doveva essere già di altezza $h$ e conteneva da sola già $2^h$ elementi
>>
>>![[Pasted image 20250326112648.png]]


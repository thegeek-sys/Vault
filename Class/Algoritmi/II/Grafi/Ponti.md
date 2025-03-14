---
Created: 2025-03-14
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
In un grafo la connessione è una proprietà che può andar persa con la perdita di un arc

![[Pasted image 20250314102426.png|center|350]]

>[!info] Definizione
>Una arco la cui eliminazione disconnette il grafo è detto **ponte**

I ponti rappresentano criticità del grafo ed è quindi utile identificarli

---
## Determinare l’insieme dei ponti del grafo
Iniziamo ricordando che il grafo può anche non avere nessun ponte (cicli) così come avere tutti i suoi archi come ponti (in un albero qualunque arco è un ponte)

![[Pasted image 20250314102655.png|center|500]]

Una prima soluzione è basata sulla ricerca esaustiva, ovvero provare per ogni arco del grafo se questo è ponte o meno.
Per verificare se un arco $(a,b)$ è un ponte per $G$ richiede $O(m)$. Infatti basterebbe eliminare l’arco $(a,b)$ da $G$ e, con una visita DFS controllare se $b$ è raggiungibile da $a$. In totale un algoritmo del genere ha complessità $m\cdot O(m)=O(m^2)=O(n^4)$

Vedremo ora che questo problema è risolvibile in $O(m)$; l’idea è quella di usare un’unica visita DFS opportunamente modificata

I ponti vanno **ricercati unicamente tra gli $n-1$ archi dell’albero DFS**. Infatti un arco non presente nell’albero DFS non può essere ponte (se lo elimino gli archi dell’albero DFS continuano a garantire la connessione)

![[Pasted image 20250314103450.png|600]]

>[!note]
>Gli archi dell’albero che non sono ponti risultano **coperti** dagli cerchi che non sono attraversati. Ovvero se hanno **archi all’indietro** (i grafi non diretti possono avere solo archi all’indietro, non in avanti e non di attraversamento)

>[!info] Proprietà
>Sia $(u,v)$ un arco dell’albero DFS con $u$ padre di $v$. L’arco $u-v$ è un ponte se e solo se non ci sono archi tra i nodi del sottoalbero radicato in $v$ e il nodo $u$ o nodi antenati di $u$
>>[!info] Dimostrazione
>>##### $\Longrightarrow$
>>![[Pasted image 20250314104039.png|150]]
>>Sia $x-y$ un arco tra un antenato di $u$ e un discendente di $v$. Dopo l’eliminazione dell’arco $u-v$ tutti i nodi dell’albero restano connessi grazie all’arco $x-y$
>>##### $\Longleftarrow$
>>![[Pasted image 20250314104058.png|150]]
>>L’eliminazione dell’arco $u-v$ disconnette i nodi dell’albero radicato in $v$ dal resto del grafo. Infatti in questo caso tutti gli archi che non appartengono all’albero e che partono da nodi del sottoalbero radicato in $v$ andrebbero verso $v$ o discendenti di $v$

Nel momento in cui l’arco $u−v$ viene “attraversato” per raggiungere il nodo $v$ da visitare, il nodo $u$ non ha informazioni per decidere se dal  sottoalbero radicato in $v$ partono archi verso suoi antenati (e quindi scoprire se $u-v$ è ponte o meno). Ma gli sarà possibile scoprirlo al termine della visita di $v$ se riceve da $v$ la giusta informazione.

### Soluzione
Per ogni arco padre-figlio $(u,v)$ presente nell’albero DFS il nodo $u$ è in grado di scoprire se l’arco $(u,v)$ è o meno un ponte usando la seguente strategia:
1. calcola la sua altezza nell’albero
2. calcola e restituisce al padre $u$ l’altezza minima che si può raggiungere con archi che partono dai nodi del sottoalbero diversi da $(u,v)$
Il nodo $u$ ricevuta l’informazione dal figlio $v$ confronta la sua altezza con quella ricevuta dal figlio. Perché l’arco sia ponte deve accadere che l’altezza di $u$ deve essere minore di quella restituita dal figlio

**Riassunto**:
- Nodo $v$ → esplora il proprio sottoalbero e restituisce al padre u il valore $b$, cioè il livello minimo raggiungibile da $v$ e i suoi discendenti utilizzando anche eventuali archi all’indietro (non passando in archi già attraversati)
- Nodo $u$ → confronta $b$ con la propria altezza. Se $b$ è maggiore dell’altezza di $u$, l’arco $(u, v)$ è l’unico collegamento e dunque è un ponte; altrimenti, c’è un percorso alternativo che collega il sottoalbero di $v$ a $u$ o ad un suo antenato, per cui l’arco non è un ponte.

![[Pasted image 20250314105406.png]]

```python
def DFS(x, padre, altezza, ponti):
	# assegna l'altezza al nodo corrente
	if padre == -1:
		altezza[x] = 0
	else:
		altezza[x] = altezza[padre]+1
	# minima altezza raggiunbile dal sottoalbero di x
	min_raggiungibile = altezza[x]
	for y in G[x]:
		# il nodo y non è stato ancora visitato
		if altezza[y] == -1:
			b = DFS(y, x, altezza, ponti)
			# l'altezza di x minore di quella ritornata da y, quindi
			# (x,y) è un ponte
			if b > altezza[x]:
				ponti.append((x,y))
			min_raggiunbile = min(min_raggiunbile, b)
		# y già visitato e (x,y) è un arco all'indietro
		elif y != padre:
			min_raggiungibile = min(min_raggiunbile, altezza[y])
	return min_raggiungibile

def trova_ponti(G):
	# array per memorizzare l'altezza dei nodi nella DFS
	altezza = [-1]*len(G)
	ponti = []
	# inizia la DFS dal nodo 0
	DFS(0, -1, altezza, ponti)
	return ponti
```
La complessità della procedura è $O(n+m)$

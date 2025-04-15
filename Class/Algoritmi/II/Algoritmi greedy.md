---
Created: 2025-04-06
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Implementazione|Implementazione]]
- [[#Assegnazione di attività|Assegnazione di attività]]
	- [[#Assegnazione di attività#Implementazione|Implementazione]]
- [[#Esercizio|Esercizio]]
---
## Introduction
Per illustrare il progetto e l’analisi di una algoritmo greedy consideriamo un problema piuttosto semplice chiamato **selezione di attività**

Abbiamo una lista di $n$ attività da eseguire:
- ciascuna attività è caratterizzata da una coppia con il suo tempo di inizio ed il suo tempo di fine
- due attività sono *compatibili* se non si sovrappongono

Vogliamo trovare un sottoinsieme di attività compatibili di massima cardinalità

>[!example]
>Istanza del problema con $n=8$
>![[Pasted image 20250406161602.png|450]]
>![[Pasted image 20250406161639.png|450]]
>
>Per le $8$ attività l’insieme da selezionare è $\{b,e,h\}$. In questo caso è facile convincersi che non ci sono altri insiemi di $3$ attività compatibili e che non c’è alcun insieme di $4$ attività compatibili. In generale possono esistere diverse soluzioni ottime

Volendo utilizzare il paradigma greedy dovremmo trovare una regola semplice da calcolare, che ci permetta di effettuare ogni volta la scelta giusta.

Per questo problema ci sono diverse potenziali regole di scelta:
- prendi l’attività compatibile che inizia prima → soluzione sbagliata
	![[Pasted image 20250406162050.png]]
- prendi l’attività compatibile che dura meno → soluzione sbagliata
	![[Pasted image 20250406162140.png]]
- prendi l’attività disponibile che ha meno conflitti con le rimanenti → soluzione sbagliata
	![[Pasted image 20250406162232.png]]

La soluzione corretta sta nel prendere sempre l’attività compatibile che **finisce prima**
![[Pasted image 20250406162338.png]]

>[!info] Dimostrazione
>Supponiamo per assurdo che la soluzione greedy $SOL$ trovata da questa regola non sia ottima. Le soluzione ottime dunque differiscono da $SOL$.
>
>Nel caso ci fossero più di una soluzione ottima prendiamo quella che differisce nel minor numero di attività da $SOL$, sia $SOL^*$. Dimostreremo ora che esiste un’altra soluzione ottima $SOL'$ che differisce ancora meno da $SOL$ che è assurdo
>
>Siano $A_{1},A_{2},\dots$ le attività nell’ordine in cui sono state scelte dal greedy e sia $A_{i}$ la prima attività scelta dal greedy e non dall’ottimo (questa attività deve essitere perché tutte le attività scartate dal greedy erano incompatibili con quelle prese dal greedy e se la soluzione avesse preso tutte le attività scelte dal greedy non potrebbe averne prese di più). 
>
>Nell’ottimo deve esserci un’altra attività $A'$ che va in conflitto con $A_{i}$ (altrimenti $SOL^*$ non sarebbe ottima in quanto potrei aggiungervi l’attività $A_{i}$). A questo punto posso sostituire in $SOL^*$ l’attività $A'$ con l’attività senza creare conflitti (perché in base alla regole di scelta greedy $A_{i}$ termina prima di $A'$).
>
>Ottengo in questo modo una soluzione ottima $SOL'$ )infatti le attività in $SOL'$ sono tutte compatibili e la cardinalità di $SOL'$ è la stessa di $SOL^*$). Ma $SOL'$ differisce da $SOL$ di un’attività in meno rispetto a $SOL^*$
>
>![[Pasted image 20250406163221.png]]

### Implementazione

```python
def selezione_a(lista):
	lista.sort(key = lambda x: x[1])
	libero = 0
	sol = []
	for inizio, fine in lista:
		if libero < inizio:
			sol.append((inizio,fine))
			libero = fine
	return sol

# >> lista = [(1,5),(14,21),(15,20),(2,9),(3,8),(6,13),(18,22),(10,13),(12,17),(16,19)]
# >> selezione_a(lista)
# [(1,5),(6,11),(12,17),(18,22)]
```
Complessità:
- ordinare la lista delle attività costa $\Theta(n\log n)$
- il $for$ viene eseguito $n$ volte e il costo di ogni iterazione è $O(1)$

Il costo totale è $\Theta(n\log n)$

---
## Assegnazione di attività
Consideriamo ora questo nuovo problema noto come **assegnazione di attività**

Abbiamo una lista di attività, ciascuna caratterizzata da un tempo di inizio ed un tempo di fine. Le attività vanno tutte eseguite e vogliamo assegnarle al minor numero di aule tenendo conto che in una stessa aula non possono eseguirsi più attività in parallelo

>[!example]
>`lista = [(1,4),(1,6),(7,8),(5,10)]`
>![[Pasted image 20250415212324.png]]

Un possibile algoritmo greedy si basa sull’idea di occupare aule finché ci sono aule da assegnare e ad ogni aula, una volta inaugurata, assegnare il maggior numero di attività non ancora assegnate che è in grado di contenere, utilizzando la funzione `soluzione_a` creata in precedenza

Ma in questo caso l’algoritmo non produce una soluzione corretta, infatti utilizzando l’esempio precedente si avrebbe
![[Pasted image 20250415213430.png]]

### Implementazione
Una soluzione alternativa sta nel selezionare ogni volta l’attività **che inizia prima** e, se è già presente nella soluzione un’aula in cui è possibile eseguirla gli viene assegnata, altrimenti si inserisce nella soluzione una nuova aula e gli viene assegnata l’attività

>[!info] Correttezza
>Sia $k$ il numero di aule utilizzate dalla soluzione. Fremo vedere che ci sono nella lista $k$ attività incompatibile a coppie e questo ovviamente implica che $k$ aule sono necessarie
>
>Sia $(a,b)$ l’attività che ha portato all’introduzione nella soluzione della $k$-esima aula. 
>In quel momento nelle altre $k-1$ aule era impossibile eseguire l’attività $(a,b)$ ma per il criterio della scelta greedy posso anche dire che nell’istante di tempo $a$ erano tutte occupate (le attività loro assegnate iniziavano prima del tempo $a$ e non sono ancora finite) quindi nell’istante di tempo $a$ a due a due tutte queste $k$ attività sono incompatibili
>
>![[Pasted image 20250415214045.png|250]]

L’idea consiste quindi in individuare efficientemente l’attività che inizia prima effettuando un pre-processing in cui ordino le attività per tempo di inizio.
Una volta fatto ciò un **heap minimo** in cui metto le coppie $(libera,i)$ dove $libera$ indica il tempo in cui si libera l’aula $i$. In questo modo in tempo $O(1)$ posso sapere qual’è l’aula che si libera prima semplicemente verificando $H[0][0]$

Se nell’aula che si libera prima si può eseguire l’attività allora dovrà assegnargliela e aggiornare il valore $libera$ della coppia che la rappresenta nell’heap. Se al contrario l’aula non può eseguire l’attività allora non sarà possibile farlo in nessuna delle altre aule e bisognerà assegnare l’attività ad una nuova aula ed inserire nell’heap la coppia che rappresenta questa nuova aula. Inserimenti e cancellazioni dall’heap costeranno $O(\log n)$

```python
def assegnazioneAule(lista):
	from heapq import heappop, heappush
	f = [[]]
	H = [(0,0)]
	lista.sort()
	
	for inizio, fine in lista:
		libera, aula = H[0]
		if libera<inizio:
			f[aula].append((inizio, fine))
			heappop(H)
			heappush(H, (fine, aula))
		else:
			f.append([(inizio, fine)])
			heappush(H, (fine, len(f)-1))
	return f
```
Complessità:
- ordinare la lista costa $\Theta(n\log n)$
- il $for$ viene eseguito $n$ volta ed all’interno del $for$ al caso pessimo può essere eseguita un’estrazione dall’heap seguita subito dopo da un inserimento nell’heap, entrambe le operazioni di costo $O(\log n)$. Il $for$ richiederà quindi tempo $O(n\log n)$

La complessità dell’algoritmo è $\Theta(n\log n)$

---
## Esercizio
Abbiamo $n$ file di dimensioni $d_{0},d_{1},\dots,d_{n-1}$ che vorremmo memorizzare su un disco di capacità $k$. Tuttavia la somma delle dimensioni di questi file eccede la capacità del disco. Vogliamo dunque selezionare un sottoinsieme degli $n$ file che abbia cardinalità massima e che possa essere memorizzato sul disco

Descrivere un algoritmo greedy che risolve il problema in tempo $\Theta(n\log n)$ e provarne la correttezza

Un algoritmo greedy per questo problema è quello che si presenta naturalmente: consideriamo i file per dimensione crescente e se c’è spazio per memorizzare il file sul disco allora facciamolo

```python
def file(D,k):
	l = [(D[i],i) for i in range(len(D))]
	l.sort()
	spazio, sol = 0, []
	for d,i in l:
		if spazio+d<=k:
			sol.append(i)
			spazio+=d
		else:
			return sol
```
La complessità dell’algoritmo è $\Theta(n\log n)$

>[!info] Correttezza
>Assumiamo per assurdo che la soluzione $sol$ prodotta dal greedy non sia ottima, devono quindi esistere insiemi con più file di $sol$ che rispettano la capacità del disco.
>Tra questi insiemi prendiamo quello con più elementi in comune con $sol$ e denotiamolo con $sol^*$
>
>Nota che:
>- esiste un file $a$ che appartiene a $sol^*$ e non a $sol$ e che occupa più spazio di qualunque file in $sol$ (questo solo perché tutti gli elementi in $sol$ occupano meno spazio di quelli non presenti in $sol$)
>- esiste un file $b$ che appartiene a $sol$ e non a $sol^*$ (questo perché $sol \not\subset sol^*$, infatti l’aggiunta di qualunque elemento a $sol$ porterebe a superare la capacità del disco)
>
>Posso dunque eliminare da $sol^*$ il file $a$ e inserire il file $b$ ottenendo un nuovo insieme di file che rispetta ancora le capacità del disco ed ha un elemento in più in comune con $sol$ contraddicendo le nostre ipotesi (il file $sol^*$ è quello con più elementi in comune con $sol$)


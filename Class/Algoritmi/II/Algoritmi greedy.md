---
Created: 2025-04-06
Class: "[[Algoritmi]]"
Related: 
Completed:
---
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


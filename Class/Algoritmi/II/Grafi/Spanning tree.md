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


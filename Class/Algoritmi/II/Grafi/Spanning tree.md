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
Per risolvere il problema del minimo albero di copertura, data la sua importanza esistono diversi algoritmi per risolvere questo problema. Ora analizzeremo **l’algoritmo di Kursal**
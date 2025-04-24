---
Created: 2025-04-24
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Recap [[Forwarding dei datagrammi IP|forwarding datagrammi IP]]
Inoltrare significa collocare il datagramma sul giusto percorso (porta di uscita del router) che lo porterà a destinazione (o lo farà avanzare verso la prossima destinazione)

Quando un host ha un datagramma da inviare lo invia al router della rete locale. Quando un router riceve un datagramma da inoltrare accede alla tabella di routing per trovare il successivo hop a cui inviarlo.
L’inoltro richiede una riga della tabella per ogni blocco di rete

---
## Introduction

>[!question] Quale percorso deve seguire un pacchetto che viene instradato da un router sorgente a un router destinazione? Se sono disponibili più percorsi, quale si sceglie?

Il routing si occupa di trovare il percorso migliore e inserirlo nella tabella di routing

>[!warning] Il routing costruisce le tabelle, il forwarding le usa

---
## Grafo di una rete di calcolatori

>[!example]
>![[Pasted image 20250424230427.png|center|370]]
>$G=(N,E)$
>$N=\text{insieme dei nodi (router)}=\{u,v,w,x,y,z\}$
>$E=\text{insieme di archi}=\{(u,v), (u,x), (v,x), (v,w), (x,w), (x,y), (w,y), (w,z), (y,z)\}$
>
>Un path nel grafo $G=(N,E)$ è una sequenza di nodi $(x_{1},x_{2},\dots,x_{n})$ tale che ognuna delle coppie $(x_{1},x_{2}),(x_{2},x_{3}),\dots,(x_{n-1},x_{n})$ sono archi di $E$

### Costi
Nel grafo $c(x,x')$ è il costo del collegamento $(x,x')$ (es. $c(w,z)=5$), dunque il costo di un cammino è semplicemente la somma di tutti i costi degli archi lungo il cammino

>[!question] Cosa rappresenta il costo?
>- lunghezza fisica del collegamento
>- velocità del collegamento
>- costo monetario per poter attraversare il collegamento

Serve quindi un **algoritmo di instradamento** per poter determinare il cammino a costo minimo

---
## Algoritmo d’instradamento con vettore distanza
L’algoritmo d’instradamento presenta  due principali caratteristiche, è:
- **distribuito** → ogni nodo riceve informazione dai vicini e opera su quelle
- **asincrono** → non richiede che tutti i nodi operino al passo con gli altri

Si basa su:
1. equazione di Bellman-Ford
2. concetto di vettore di distanza

### Equazione di Bellman-Ford
Definisce $D_{x}(y):=\text{il costo del percoso a costo minimo dal nodo y al nodo y}$

Allora
$$
D_{x}(y)=\text{min}_{v}\{c(x,v)+D_{v}(y)\}
$$
dove $\text{min}_{v}$ riguarda tutti i vicini di $x$

#### Rappresentazione grafica
I percorsi $a\to b$, $b\to y$, $c\to y$ sono percorsi a costo minimo precedentemente stabiliti e $x\to y$ è un nuovo percorso a costo minimo

![[Pasted image 20250424234155.png|450]]
$$
D_{xy}=\text{min}\{(c_{xa}+D_{ay}),(c_{xb}+D_{by}),(c_{xc}+D_{cy})\}
$$

### Vettore distanza
Un albero a costo minimo è una combinazione di percorsi a costo minimo dalla radice dell’albero verso tutte le destinazioni

Il vettore di distanza p un array monodimensionale che rappresenta l’albero. Un vettore di distanza non fornisce il percorso da seguire per giungere alla destinazione ma solo i costi minimi per le destinazioni

---
## Come viene creato il vettore delle distanze?
Ogni nodo della rete quando viene inizializzato crea un vettore distanza iniziale con le informazioni che riesce ad ottenere dai propri vicini (nodi a cui è direttamente collegato)

Per creare il vettore dei vicini invia messaggi di `hello` attraverso le sue interfacce (e lo stesso fanno i vicini) e scopre l’identità dei vicini e la sua distanza da ognuno di essi
Dopo che ogni nodo ha creato il suo vettore ne invia una copia ai suoi vicini

>[!example]
>Vettori distanza iniziali dopo messaggi di hello
>![[Pasted image 20250424235444.png|400]]
>
>Quando un nodo riceve un vettore distanza un vicino provvede ad aggiornare il suo vettore distanza applicando l’equazione di Bellman-Ford
>
>Cosa succede quando $B$ riceve una copia di $A$?
>![[Pasted image 20250424235730.png|300]]
>
>Cosa succede se ora $B$ riceve una copia di $E$?
>![[Pasted image 20250424235818.png|300]]

---
## Algoritmo con vettore distanza
L’idea di base è che ogni nodo invia una copia del proprio vettore distanza a ciascuno dei suoi vicini. Quando un nodo $x$ riceve un nuovo vettore distanza $DV$ da qualcuno dei suoi vicini, lo salva e usa la formula Bellman-Ford per aggiornare il proprio vettore distanza come segue:
$$
D_{x}(y)\gets \text{min}_{v}(c(x,v)+D_{v}(y))
$$
per ciascun nodo $y$ in $N$

Se il vettore distanza del nodo $x$ è cambiato per via di tale passo di aggiornamento, il nodo $x$ manderà il proprio vettore distanza aggiornato a ciascuno dei suoi vicini, i quali a loro volta aggiornano il loro vettore distanza


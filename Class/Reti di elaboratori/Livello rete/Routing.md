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

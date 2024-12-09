---
Created: 2024-12-09
Class: 
Related: 
Completed:
---
---
## Introduction
Il **deadlock** (o stallo) è un blocco permanente di un certo insieme di processi che competono per delle risorse di sistema o comunicano tra loro.
Il motivo di base del deadlock è la **richiesta contemporanea delle stesse risorse** da parte di due o più processi.

![[Pasted image 20241209220045.png|center|480]]

Nonostante tutto non esiste una soluzione universale per risolvere questo problema, bisogna infatti analizzare caso per caso e risolverlo in una maniera opportuna.

## Joint progress diagram
Quando ci troviamo di fronte ad un deadlock tra due processi, questo può essere analizzato attraverso questo semplice diagramma

![[Pasted image 20241209220247.png|600]]

>[!info] I due processi richiedono la risorsa successiva prima di aver rilasciato quella che stanno usando

>[!note]
>Linee orizzontali → momento in cui `P` è in esecuzione
>Linee verticali → momento in cui `Q` è in esecuzione
>Quando intercettano un quadrante vuol dire che viene eseguita l’operazione indicata (es. Get A)
>
>5 non va in deadlock, infatti prova a richiedere una risorsa già occupata, quindi il processo viene mezzo in blocked


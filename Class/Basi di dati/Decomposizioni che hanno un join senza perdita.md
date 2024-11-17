---
Created: 2024-11-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Quando si decompone uno schema occorre tenere presente (oltre al dover mantenere tutte le dipendenze originali), deve permettere di **ricostruire mediante il join naturale** ogni **istanza legale dello schema originario** (senza aggiunta di tuple estranee)
[[Terza forma normale#La 3NF non basta#Esempi|Qui]] qualche esempio in cui ciò non accade

---
## Definizione
Se si decompone uno schema di relazione $R$ si vuole che la decomposizione $\{R_{1},R_{2},\dots,R_{k}\}$ ottenuta sia tale che ogni istanza legale $r$ di $R$ sia ricostruibile mediante join naturale ($\bowtie$) da un’istanza legale $\{r_{1},r_{2},\dots,r_{k}\}$ dello schema decomposto $\{R_{1},R_{2},\dots,R_{k}\}$

>[!info] Definizione
>Sia $R$ uno schema di relazione. Una decomposizione $\rho=\{R_{1},R_{2},\dots,R_{k}\}$ di $R$ ha un **join senza perdita** se per ogni istanza legale $r$ di $R$ si ha $r=\pi_{R_{1}}(r)\bowtie \pi_{R_{2}}(r)\bowtie\dots \bowtie \pi_{R_{k}}(r)$

---
## Teorema
Sia $R$ uno schema di relazione e $\rho=\{R_{1},R_{2},\dots ,R_{k}\}$ una decomposizione di $R$. Per ogni istanza legale $r$ di $R$, indicato con $m_{\rho}(r)=\pi_{R1}(r)\bowtie \pi_{R2}(r)\bowtie\dots \bowtie \pi_{Rk}(r)$ si ha:
- $r \subseteq m_{\rho}(r)$
- $\pi_{R_{i}}(m_{\rho}(r))=\pi_{R_{i}}(r)$
- $m_{\rho}(m_{\rho}(r))=m_{\rho}(r)$

---
## Digressione
Consideriamo la solita istanza legale di $R=ABC$ con l’insieme di dipendenze funzionali $F=\{A\to B,C\to B\}$ (non è in 3NF - $B$ non è contenuto in una chiave)

![[Pasted image 20241117185711.png|250]]

In base alle possibili decomposizioni dello schema, questa istanza si decompone in
![[Pasted image 20241117185753.png|350]]
La prima è ottenuta proiettando l’istanza originale su $AB$. La seconda è ottenuta proiettando l’istanza originale su $BC$.

>[!note]
>Notare l’eliminazione del duplicato. Forse questa eliminazione ci farà perdere tuple originali? No

Dovrebbe essere possibile ricostruire l’istanza di partenza esattamente tramite join invece se si effettua il join delle due istanze legali risultanti dalla decomposizione si ottiene

![[Pasted image 20241117190130.png]]

Occorre garantire che il join delle istanze risultati dalla decomposizione non riveli *perdita di informazioni*

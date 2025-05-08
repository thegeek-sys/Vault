---
Created: 2025-05-07
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Struttura di internet
![[Pasted image 20250507215644.png|center|500]]

Gli ISP forniscono servizi a livelli differenti:
- **dorsali** → gestite da società private di telecomunicazioni, che forniscono la connettività globale (connesse tramite peering point)
- **network provider** → che utilizzano le dorsali per avere connettività globale e forniscono connettività ai clienti Internet
- **customer network** → che usano i servizi  dei network provider

### Impossibilità di usare un singolo protocollo di routing
Abbiamo fin qui visto la rete come una collezione di router interconnessi in cui ogni router era indistinguibile dagli altri

Però nella pratica si hanno 200 milioni di destinazioni e archiviare le informazioni d’instradamento su ciascun host richiederebbe un’enorme quantità di memoria, il traffico generato dagli aggiornamenti link state non lascerebbero banda per i pacchetti di dati e il distance vector non convergerebbe mai

Per questi motivi è necessaria **autonomia amministrativa** per cui ciascuno dovrebbe essere in grado di amministrare la propria rete nel modo desiderato, pur mantenendo la possibilità di connetterla alle reti esterne

---
## Instradamento gerarchico
Ogni ISP è un **sistema autonomo** (AS, *autonomous system*) e ogni AS può eseguire un protocollo di routing che soddisfa le proprie esigenze. I router di uno stesso AS eseguono lo stesso algoritmo di routing (si parla di protocollo di routing interno al sistema autonomo - **intra-AS** - o intradominio, o interior gateway protocol - IGP), ma i router appartenenti a differenti AS possono eseguire protocolli di instradamento intra-AS diversi

E’ quindi necessario avere un solo protocollo inter-dominio che gestisce il routing tra i vari AS. In questo caso si parla di protocollo di routing inter-AS o inter-dominio, o exterior gateway protocol (EGP)

Per **router gateway** si intendono i router che connettono gli AS tra loro e che hanno il compito aggiuntivo di inoltrare pacchetti a destinazioni interne

---
## Sistemi autonomi
Ogni ISP è un sistema autonomo, e ad ogni AS viene assegnato un numero identificativo univoco di $16$ bit (**autonomous number** - *ASN*) dall’ICANN

Gli AS possono avere diverse dimensioni e sono classificati in base al modo in cui sono connessi ad altri AS:
- **AS stub** → ha un solo collegamento verso un altro AS; il traffico è generato o destinato allo stub ma non transita attraverso di esso (es. grande azienda)
- **AS multihomed** → ha più di una connessione con altri AS ma non consente transito di traffico (azienda che usa servizi di più di un network provider ma non fornisce connettività agli altri AS)
- **AS di transito** → è collegato a più AS e consente il traffico (network provider e dorsali)

>[!example] Instradamento inter-AS
>![[Pasted image 20250508101453.png]]
>Ogni router all’interno degli AS sa come raggiungere tutte le reti che si trovano nel suo AS ma non sa come raggiungere una rete che si trova in un altro AS

### Sistemi autonomi interconnessi
Ciascun sistema autonomo sa come inoltrare pacchetti lungo il percorso ottimo verso qualsiasi destinazione interna al gruppo

![[Pasted image 20250508100851.png|550]]
- il sistema AS1 ha quattro router
- i sistemi AS2 e AS3 hanno tre router ciascuno
- i protocolli d’instradamento dei tre sistemi autonomi non sono necessariamente gli stessi
- i router 1b, 1c, 2a e 3a sono gateway (devono eseguire un protocollo aggiuntivo)

#### Routing intra-dominio
- RIP → Routing Information Protocol
- OSPF → Open Shortest Path First

#### Routing inter-dominio
- BGP → Border Gateway Protocol

---
## Border Gateway Protocol
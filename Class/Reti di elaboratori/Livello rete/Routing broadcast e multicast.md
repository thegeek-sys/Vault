---
Created: 2025-05-08
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Index
- [[#Unicast|Unicast]]
- [[#Broadcast|Broadcast]]
	- [[#Broadcast#Uncontrolled flooding|Uncontrolled flooding]]
	- [[#Broadcast#Controlled flooding|Controlled flooding]]
		- [[#Controlled flooding#Sequence number|Sequence number]]
		- [[#Controlled flooding#Reverse path forwarding (RPF)|Reverse path forwarding (RPF)]]
	- [[#Broadcast#Spanning tree (center-based)|Spanning tree (center-based)]]
		- [[#Spanning tree (center-based)#Broadcast|Broadcast]]
- [[#Multicast|Multicast]]
	- [[#Multicast#Confronto tra multicast e unicast multiplo|Confronto tra multicast e unicast multiplo]]
	- [[#Multicast#Indirizzamento multicast|Indirizzamento multicast]]
	- [[#Multicast#Indirizzi multicast|Indirizzi multicast]]
	- [[#Multicast#Gruppi multicast|Gruppi multicast]]
	- [[#Multicast#Protocolli multicast|Protocolli multicast]]
		- [[#Protocolli multicast#IGMP|IGMP]]
	- [[#Multicast#Problema del routing multicast|Problema del routing multicast]]
		- [[#Problema del routing multicast#Approcci per determinare l’albero di instradamento multicast|Approcci per determinare l’albero di instradamento multicast]]
	- [[#Multicast#Instradamento multicast in Internet|Instradamento multicast in Internet]]
---
## Unicast
Per routing **unicast** si intende la comunicazione tra una sorgente e una destinazione (IP sorgente → IP destinazione)

![[Pasted image 20250508122047.png|550]]

---
## Broadcast
Si parla di routing **broadcast** quando si vuole inviare un pacchetto da un nodo sorgente a tutti i nodi della rete. Ovvero una comunicazione $1$ a $N$, dove $N$ sono tutti i nodi della rete (IP sorgente → indirizzo broadcast)

Esistono due possibilità per eseguire il broadcast:
- **uncontrolled flooding**
- **controlled flooding**
	- sequence number
	- reverse path forwarding

### Uncontrolled flooding
Con l’uncontrolled flooding quando un nodo riceve un pacchetto broadcast, lo duplica e lo invia a tutti i nodi vicini (eccetto a quello da cui lo ha ricevuto)

![[Pasted image 20250508122451.png|150]]

In questo caso se il grafo ha cicli, una o più copie del pacchetto cicleranno all’infinito nella rete

### Controlled flooding
#### Sequence number
Qui non vengono forwardati i pacchetti già ricevuti e inoltrati

In particolare gni nodo tiene una lista di $(\text{indirizzo IP}, \#seq)$ dei pacchetti già ricevuti, duplicati, inoltrati. Quando riceve un pacchetto controlla nella lista, se già inoltrato lo scarta, altrimenti lo forwarda

#### Reverse path forwarding (RPF)
Forwarda il pacchetto se e solo se è arrivato dal link che è sul suo shortest path (unicast) verso la sorgente

![[Pasted image 20250508122803.png|350]]

Viene in questo modo eliminato il problema di inondare la rete con troppi pacchetti. Però l’RPF non elimina completamente la trasmissione di pacchetti ridondanti

>[!example]
>$B$, $C$, $D$, $E$, $F$ ricevono uno o due pacchetti ridondati

Ma ogni pacchetto dovrebbe ricevere una sola copia del pacchetto broadcast, la soluzione sta nel costruire uno **spanning tree** prima di inviare i pacchetti broadcast

### Spanning tree (center-based)
Una volta preso come centro un nodo ($E$), ogni nodo invia un messaggio di join in unicast verso il centro

I messaggi vengono inoltrati finché arrivano o a un nodo che già appartiene all’albero o alla radice ($E$)

![[Pasted image 20250508124954.png]]

#### Broadcast
I pacchetti vengono inoltrati solo sui link dell’albero, ogni nodo riceve solo una copia del pacchetto

![[Pasted image 20250508125117.png]]

---
## Multicast
Si parla di routing **broadcast** quando si ha una comunicazione tra una sorgente e un gruppo di destinazioni

![[Pasted image 20250508125201.png]]

### Confronto tra multicast e unicast multiplo
![[Pasted image 20250508125308.png]]

Nel caso di multicast si ha un solo datagramma alla sorgente che viene duplicato dai router, mentre l’unicast multiplo risulta essere inefficiente e aggiunge ritardi

### Indirizzamento multicast
Molte applicazioni richiedono il trasferimento di pacchetti da uno o più mittenti ad un gruppo di destinatari:
- trasferimento di un aggiornamento SW su un gruppo di macchine
- streaming (audio/video) ad un gruppo di utenti o studenti
- applicazioni con dati condivisi (lavagna elettronica condivisa da più utenti)
- aggiornamento di dati (andamento di borsa)
- giochi multi-player interattivi

>[!question] Come è possibile comunicare con host che partecipano ad un gruppo ma appartengono a reti diverse?
>L’indirizzo di destinazione nell’IP può essere uno solo, quindi la soluzione è di usare un indirizzo per tutto il gruppo, il cosiddetto **indirizzo multicast**

>[!example]
>![[Pasted image 20250508125721.png]]
>
>I router devono sapere quali host sono associati ad un gruppo multicast

### Indirizzi multicast
Il blocco di indirizzi riservati per il multicast in IPv4 è `224.0.0.0/4` (ovvero da `224.0.0.0` a `239.255.255.255`) per un totale di $2^{28}$ gruppi

### Gruppi multicast
L’appartenenza ad un gruppo non ha alcuna relazione con il prefisso associato alla rete. Infatti un host che appartiene ad un gruppo ha un indirizzo multicast separato e aggiuntivo rispetto all primario
L’appartenenza non è un attributo fisso dell’host (il periodo di appartenenza può essere limitato)

![[Pasted image 20250509110428.png|350]]
Un router inoltre deve scoprire quali gruppi sono presenti in ciascuna delle sue interfacce per poter propagare le informazioni agli altri router

### Protocolli multicast
Per il multicast sono necessari due protocolli:
- per raccogliere le informazioni di appartenenza ai gruppi
- per diffondere le informazioni di appartenenza

#### IGMP
L’**Internet Group Management Protocol** (*IGMP*) è un protocollo che lavora tra un host e il router che gli è direttamente connesso e offre agli host il mezzo di informare i router ad essi connessi del fatto che un’applicazione in esecuzione vuole aderire ad uno specifico gruppo multicast

![[Pasted image 20250509110843.png]]

I messaggi in questo caso sono incapsulati in datagrammi IP, con IP protocol number 2 (sono mandati con TTL a 1)

Messaggi IGMP:
- `membership query` ($\text{router}\to \text{host}$) → per determinare a quali gruppi hanno aderito gli host su ogni interfaccia (inviati periodicamente)
- `membership report` ($\text{host}\to \text{router}$) → per informare il router su un’adesione, anche non in seguito a una query (al momento dell’adesione)
- `leave group` ($\text{host}\to \text{router}$) → quando si lascia un gruppo (è opzionale, il router può capire che non ci sono host associati a quel gruppo quando non riceve report in risposta a query)

Un router multicast tiene una lista per ciascuna sottorete dei gruppi multicast con un timer per membership (la membership può essere aggiornata o da report inviati prima della scadenza del timer oppure tramite messaggi di leave espliciti)

### Problema del routing multicast
Fra la popolazione complessiva di router solo alcuni (quelli collegati a host del gruppo multicast). E’ quindi necessario un protocollo che coordini i router multicast in Internet (instradare pacchetti multicast dalla sorgente alla destinazione)

>[!example]
>$A$, $B$, $E$, $F$ sono router che devono ricevere il traffico multicast
>![[Pasted image 20250509122958.png|350]]
>
>L’obiettivo è quello di trovare un albero che colleghi tutti i router connessi ad host che appartengono al gruppo multicast. I pacchetti verranno instradati su questo albero

#### Approcci per determinare l’albero di instradamento multicast
Si hanno due possibili approcci:
- **albero condiviso dal gruppo**
- **albero basato sull’origine**

![[Pasted image 20250509123415.png|center|300]]
Nell’albero condiviso dal gruppo viene costruito un singolo albero d’instradamento condiviso da tutto il gruppo multicast in cui un router agisce da rappresentante del gruppo
Se il mittente del traffico multicast non è il centro, allora esso invierà il traffico in unicast al centro, e il centro provvederà a inviarlo al gruppo

![[Pasted image 20250509123442.png|center|300]]
Nell’albero basato sull’origine viene creato un albero per ciascuna origine nel gruppo multicast dunque ci sono tanti alberi quanti sono i mittenti nel gruppo multicast
Per la costruzione si usa un algoritmo basato su reverse path forwarding, con pruning (potatura)

### Instradamento multicast in Internet
*Intra-dominio* multicast (interno a un sistema autonomo)
- **DVMRP** → distance-vector multicast routing protocol
- **MOSPF** → multicast open shortest path first
- **PIM** → protocol independent multicast

*Inter-dominio* multicast (tra sistemi autonomi)
- **MBGP** → multicast border gateway protocol


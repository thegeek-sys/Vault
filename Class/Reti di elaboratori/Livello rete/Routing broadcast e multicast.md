---
Created: 2025-05-08
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
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


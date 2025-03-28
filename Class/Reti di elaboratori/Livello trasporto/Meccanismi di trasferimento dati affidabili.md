---
Created: 2025-03-27
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Index
- [[#Stop-and-wait|Stop-and-wait]]
	- [[#Stop-and-wait#Numeri di sequenza e riscontro|Numeri di sequenza e riscontro]]
	- [[#Stop-and-wait#FSM mittente|FSM mittente]]
	- [[#Stop-and-wait#FSM destinatario|FSM destinatario]]
	- [[#Stop-and-wait#Diagramma di comunicazione|Diagramma di comunicazione]]
	- [[#Stop-and-wait#Efficienza|Efficienza]]
- [[#Protocolli con pipeline|Protocolli con pipeline]]
- [[#Go back N|Go back N]]
	- [[#Go back N#Numeri di sequenza e riscontro|Numeri di sequenza e riscontro]]
	- [[#Go back N#Finestra di invio|Finestra di invio]]
	- [[#Go back N#Finestra di ricezione|Finestra di ricezione]]
	- [[#Go back N#Timer e rispedizione|Timer e rispedizione]]
	- [[#Go back N#FSM mittente|FSM mittente]]
	- [[#Go back N#FSM destinatario|FSM destinatario]]
	- [[#Go back N#Diagramma di comunicazione|Diagramma di comunicazione]]
		- [[#Diagramma di comunicazione#Ack cumulativo|Ack cumulativo]]
		- [[#Diagramma di comunicazione#Perso pacchetto dati|Perso pacchetto dati]]
	- [[#Go back N#Dimensione della finestra di invio|Dimensione della finestra di invio]]
- [[#Ripetizione selettiva|Ripetizione selettiva]]
	- [[#Ripetizione selettiva#Schema generale|Schema generale]]
	- [[#Ripetizione selettiva#Finestra di invio e ricezione|Finestra di invio e ricezione]]
	- [[#Ripetizione selettiva#Timer e riscontri|Timer e riscontri]]
	- [[#Ripetizione selettiva#FMS mittente|FMS mittente]]
	- [[#Ripetizione selettiva#FSM destinatario|FSM destinatario]]
	- [[#Ripetizione selettiva#Diagramma di comunicazione|Diagramma di comunicazione]]
	- [[#Ripetizione selettiva#Dimensione delle finestre di invio e ricezione|Dimensione delle finestre di invio e ricezione]]
- [[#Protocolli bidirezionali: piggybacking|Protocolli bidirezionali: piggybacking]]
- [[#Riassunto dei meccanismi|Riassunto dei meccanismi]]
---
## Stop-and-wait
Lo **stop-and-wait** è un meccanismo di trasferimento dati orientato alla connessione con controllo di flusso e controllo degli errori

In questo caso mittente e destinatario per comunicare usano una **finestra scorrevole di dimensione $1$**. Il mittente invia un pacchetto alla volta e ne attende l’*ack* prima di spedire il successivo.

Quando il pacchetto arriva al destinatario viene calcolato il checksum. In caso il checksum corrisponda viene inviato l’ack al mittente, ma in caso contrario il pacchetto viene scartato senza informare il mittente. Infatti, per capire se un pacchetto è andato perso il mittente usa un **timer**; una volta che è scaduto il timer senza ricevere ack viene rinviato il pacchetto

![[Pasted image 20250327233134.png|center|650]]

Il mittente deve tenere una copia del pacchetto spedito finché non riceve riscontro

### Numeri di sequenza e riscontro
Per gestire i pacchetti duplicati lo stop&wait utilizza i numeri di sequenza. Per fare ciò si vuole identificare l’intervallo più piccolo possibile che possa consentire la comunicazione senza ambiguità.

Supponiamo che il mittente abbia inviato il pacchetto con numero di sequenza $x$. Si possono verificare 3 casi:
1. Il pacchetto arriva correttamente al destinatario che invia un riscontro. Il riscontro arriva al mittente che invia il pacchetto successivo numerato $x+1$
2. Il pacchetto risulta corrotto o non arriva al destinatario. Il mittente allo scadere del timer invia nuovamente il pacchetto $x$
3. Il pacchetti arriva correttamente al destinatario ma il riscontro viene perso o corrotto. Scade il timer e il mittente rispedisce $x$. *Il destinatario riceve un duplicato, se ne accorge?*

I numeri di sequenza $0$ e $1$ sono sufficienti per il protocollo stop and wait.
Come convenzione infatti si è scelto che il numero di riscontro (ack) indica il numero di sequenza del prossimo pacchetto atteso dal destinatario (pacchetto che deve arrivare).

>[!example]
>Se il destinatario ha ricevuto correttamente il pacchetto $0$ invia un riscontro con valore $1$ (che significa che il prossimo pacchetto atteso ha numero di sequenza $1$)

>[!hint]
>Nel meccanismo stop and wait, il numero di riscontro indica, in aritmetica modulo 2, il numero di sequenza del prossimo pacchetto atteso dal destinatario

### FSM mittente
![[Pasted image 20250327234050.png]]
Una volta inviato un pacchetto, il mittente si blocca (non spedisce pacchetto successivo) e aspetta (stop and wait) finché non riceve ack

### FSM destinatario
![[Pasted image 20250327234149.png]]
Il destinatario è sempre nello stato ready

### Diagramma di comunicazione
![[immagine_sfondo_bianco.png]]

### Efficienza
Consideriamo il prodotto $\text{rate}\cdot \text{ritardo}$ (misura del numero di bit che il mittente può inviare prima di ricevere un ack, volume della pipe in bit). Se il rate è elevato e il ritardo consistente allora lo stop and wait sarà inefficiente

>[!example]
>In un sistema che utilizza stop and wait abbiamo:
>- rate → $1 \text{ Mbps}$
>- ritardo di andata e ritorno di $1\text{ bit}$ → $20\text{ ms}$
>
>Quanto vale $\text{rate}\cdot \text{ritardo}$?
>Se i pacchetti hanno dimensione $1000\text{ bit}$, qual è la percentuale di utilizzo del canale
>
>$\text{rate}\cdot \text{ritardo}=(1\times 10^6)\times(20\times 10^{-3})=20000\text{ bit}$
>Il mittente potrebbe inviare $200000\text{ bit}$ nel tempo necessario per andare dal mittente al ricevente e viceversa ma ne invia solo $1000$
>
>Il **coefficiente di utilizzo** del canale è $\frac{1000}{200000}=5\%$, risultando molto inefficiente

---
## Protocolli con pipeline
Tramite il **pipelining**, il mittente ammette più pacchetti in transito ancora da notificare. Ciò però comporta due principali modifiche del modello precedente:
- l’intervallo dei numeri di sequenza deve essere incrementato
- buffering dei pacchetti presso il mittente e/o ricevente

![[Pasted image 20250327235619.png]]

Esistono due forme generiche di meccanismi con pipeline:
- **Go-back-N**
- **ripetizione selettiva**

---
## Go back N
Lo schema generale prevede che possano essere mandati più pacchetti nonostante questi non siano stati ancora ricevuti

![[Pasted image 20250327235853.png|center|650]]

### Numeri di sequenza e riscontro
I numeri di sequenza sono calcolati modulo $2^m$ dove $m$ è la dimensioen del campo “numero di sequenza” in bit.

Ack (come nel precedente) indica il numero di sequenza del prossimo pacchetto atteso. Però in questo caso si tratta di un **ack comulativo** che indica che tutti i pacchetti fino al numero di sequenza indicato nell’ack sono stati ricevuti correttamente

>[!example]
>$\text{AckNo}=7$ → i pacchetti fino al $6$ sono stati ricevuti correttamente e il destinatario attende il $7$

### Finestra di invio
![[Pasted image 20250328000419.png|center|550]]

>[!info]
>La finestra di invio è un concetto astratto che definisce una porzione immaginaria di dimensione massima $2^m-1$ con tre variabili, $S_{f}$, $S_{n}$, $S_{\text{size}}$

La finestra di invio può scorrere uno o più posizioni quando viene un riscontro privo di errori con $\text{ackNo}$ maggiore o oguale a $S_{f}$ e, minore di $S_{n}$ in aritmetica modulare

>[!example]
>![[Pasted image 20250328000712.png|550]]

### Finestra di ricezione
La finestra di ricezione ha dimensione $1$, infatti il destinatario è sempre in attesa di uno specifico pacchetto, qualsiasi pacchetto arrivato fuori sequenza (appartenente alle due regioni esterne all finestra) viene scartato

![[Pasted image 20250328000836.png|center|550]]

La finestra di ricezione può scorrere di una sola posizione: $R_{n}=(R_{n}+1)\text{ mod }2^m$

### Timer e rispedizione
Il mittente mantiene un timer per il più vecchio pacchetto non riscontrato. Allo scadere del timer, Go back N, vengono rispediti tutti i pacchetti a partire dal più vecchio non riscontrato

In questo caso il destinatario è pronto ricevere un solo pacchetto con un numero di sequenza determinato. Se salta il pacchetto, ma ne vengono ricevuti altri successi, bisognerà rimandare tutti i pacchetti a partire dal primo non ricevuto

>[!example]
>$S_{f}=3$ è il mittente ha inviato il pacchetto $6$ ($S_{n}=7$). Scade il timer, allora i pacchetti $3,4,5,6$ nono sono stati riscontrati e devono essere rispediti

### FSM mittente
![[Pasted image 20250328001429.png]]

### FSM destinatario
![[Pasted image 20250328001611.png]]

### Diagramma di comunicazione
#### Ack cumulativo
![[Pasted image 20250328001758.png]]

#### Perso pacchetto dati
![[Pasted image 20250328002027.png]]

### Dimensione della finestra di invio
Quale è la relazione fra lo spazio dei numeri di sequenza e la dimensione della finestra di invio? Possiamo avere una finestra di dimensione $2^m$

![[Pasted image 20250328002549.png|550]]
In questo caso nonostante venga rinviato il pacchetto zero poiché è perso $\text{ack}0$, il destinatario pensa di aver ricevuto pacchetto iniziale di una nuova sequenza e quindi fa scorrere la finestra

Per risolvere questo problema la dimensione deve essere $2^m -1$
![[Pasted image 20250328002848.png|450]]

---
## Ripetizione selettiva
In Go back N per un solo pacchetto perso si trasmettono tutti i successivi già inviati nel pipeline. Ciò potrebbe essere sconveniente soprattutto nel caso in cui in rete ci sia un congestione, infatti la rispedizione di tutti i pacchetti peggiora la congestione

Nella **ripetizione selettiva**, il mittente ritrasmette soltanto i pacchetti per i quali non ha ricevuto un ack (un timer del mittente per ogni pacchetto non riscontrato). Quindi il ricevente invia **riscontri specifici** per tutti i pacchetti ricevuti correttamente (sia in ordine, sia fuori sequenza) e se necessario un buffer dei pacchetti per eventuali consegne in sequenza al livello superiore

### Schema generale
![[Pasted image 20250328003421.png|650]]

### Finestra di invio e ricezione
La finestra di invio e ricezione hanno la stessa dimensione (a differenza del precedente)

![[Pasted image 20250328003638.png]]

### Timer e riscontri
Selective repeat usa **un timer per ogni pacchetto** in attesa di riscontro (quando scade un timer si rinvia solo il relativo pacchetto). Dunque si tratta di un **riscontro invidivuale** ovvero associato al singolo pacchetto; il numero di riscontro indica il numero di sequenza di un pacchetto ricevuto correttamente (non il prossimo atteso, in questo caso non ha alcun senso)

### FMS mittente
![[Pasted image 20250328005951.png]]

### FSM destinatario
![[Pasted image 20250328010024.png]]

### Diagramma di comunicazione
![[Pasted image 20250328010109.png]]

>[!warning]
>I pacchetti possono essere consegnati al livello applicazione se:
>- è stato ricevuto un insieme di pacchetti consecutivi
>- l’insieme deve partire dall’inizio della finestra

### Dimensione delle finestre di invio e ricezione
Con le finestre $2^m-1$ potrebbe accadere che gli ack si perdano e nonostante ciò, quando il mittente rispedisce i pacchetti, questi vengono accettati da destinatario e interpretati come nuovi pacchetti della nuova sequenza

![[Pasted image 20250328010542.png|500]]

Usando come dimensione $2^{m-1}$ invece questo problema non si pone poiché quando viene rispedito un pacchetto, ormai la finestra del destinatario sarà slidata quindi non sarò più possibile ricevere quel pacchetto

![[Pasted image 20250328010749.png|500]]

---
## Protocolli bidirezionali: piggybacking
Abbiamo mostrato fino ad ora meccanismi unidirezionali in cui i pacchetti vengono mandati in una direzione e gli ack nella direzione opposta.
Per migliorare l’efficienza dei protocolli bidirezionali viene utilizzata la tecnica del **piggybacking**: quando un pacchetto trasporta dati da $A$ a $B$, può trasportare anche i riscontri relativi ai pacchetti ricevuti da $B$ e viceversa

---
## Riassunto dei meccanismi

| Meccanismo                      | Uso                           |
| ------------------------------- | ----------------------------- |
| Checksum                        | Per gestire errori nel canale |
| Ackknowledgement                | Per gestire errori nel canale |
| Timeout                         | Perdita pacchetti             |
| Finestra scorrevole, pipelining | Maggior utilizzo della rete   |

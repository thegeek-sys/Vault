---
Created: 2025-05-17
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Introduction
Esistono diversi tipi di reti wireless:
- LAN wireless → disponibili in campus universitari, uffici, bar, aree pubbliche
- reti cellulari → il numero di abbonati alla telefonia mobile supera quello della telefonia fissa
- bluetooth
- reti di sensori, RFID, smart ojects

Ecco alcuni degli standard esistenti
![[Pasted image 20250517105948.png]]

Quello più famoso è lo standard IEEE:

| Protocol | Release date | Freq.     | Rate (typical) | Rate (max) | Range (indoor) |
| -------- | ------------ | --------- | -------------- | ---------- | -------------- |
| Legacy   | 1997         | 2.4 GHz   | 1 Mbps         | 2 Mbps     | ?              |
| 802.11a  | 1999         | 5 GHz     | 25 Mbps        | 54 Mbps    | ~30 m          |
| 802.11b  | 1999         | 2.4 GHz   | 6.5 Mbps       | 11 Mbps    | ~30 m          |
| 802.11g  | 2003         | 2.4 GHz   | 25 Mbps        | 54 Mbps    | ~30 m          |
| 802.11n  | 2008         | 2.4/5 GHz | 200 Mbps       | 540 Mbps   | ~50 m          |

---
## Elementi
Una LAN wireless è composta da:
- *wireless host* → utilizzato per eseguire applicazioni e può essere stazionario o mobile (wireless non vuol dire necessariamente mobile)
- *base station* → tipicamente connesso ad una rete cablata; rasponsabile di mandare pacchetti tra la rete cablata e gli host wireless nella propria “area”
- *wireless link* → tipicamente usato per connettere gli host alla base station (usato anche come collegamento con il backbone); i protocolli ad accesso multiplo regolano l’accesso al link ed ha vari rates e range di trasmissione

![[Screenshot 2025-05-17 at 11.09.15.png]]

---
## Caratteristiche
Si hanno numerose differenze rispetto ad una classica rete calbata:
- mezzo trasmissivo → aria, segnale broadcast, mezzo condiviso dagli host della rete
- host wireless → non è fisicamente connesso alla rete e può muoversi liberamente
- connessione ad altre reti → mediante una stazione base detta Access Point (AP) che unisce l’ambiente wireless all’ambiente cablato

### Migrazione dall’ambiente cablato al wireless
Il funzionamento di una rete cablata o wireless dipende dai due sottolivelli inferiori dello stack protocollare (collegamento e fisico)

Per migrare dalla rete cablata a quella wireless è sufficiente cambiare le schede di rete e sostituire lo switch di collegamento con un AP (gli indirizzi MAC cambiano mentre gli IP rimangono gli stessi)

---
## Reti ad hoc
Le **reti ad hoc** (senza infrastruttura, senza un hotspot) sono composte da un insieme di host che si auto-organizzano per formare una rete e comunicano liberamente tra di loro (ogni host deve eseguire le funzionalità di rete quali network setup, routing, forwarding, etc.)

![[Pasted image 20250517112420.png|250]]

---
## Caratteristiche del link wireless
### Attenuazione del segnale
La forza dei segnali elettromagnetici diminuisce rapidamente all’aumentare della distanza dal trasmettitore in quanto il segnale si disperde in tutte le direzioni, fino ad arrivare al punto in cui la trasmissione non è più decodificabile

![[Pasted image 20250517112628.png|400]]

### Propagazione multi-path
Quando un’onda radio trova un ostacolo, tutta o una parte dell’onda è riflessa, con una perdita di potenza
Un segnale sorgente può arrivare, tramite riflessi successivi (su muri, terreno, oggetti), a raggiungere una stazione o un punto di accesso attraverso percorsi multipli

![[Pasted image 20250517124218.png|350]]

### Interferenze
Si possono avere due tipi di interferenze:
- dalla stessa sorgente → un destinatario può ricevere più segnali dal mittente desiderato a causa del multipath
- da altre sorgenti → altri trasmettitori stanno usando la stessa banda di frequenza per comunicare con altri destinatari

---
## Errori
Le caratteristiche dei link wireless causano errori. Il tasso di errore è misurato con il *Signal to Noise Radio* (**SNR**, o rapporto segnale-rumore) che misura il rapporto tra il segnale buono e il rumore esterno

Se l’SNR è alto, il segnale è più forte del rumore, quindi può essere convertito in dati reali, se invece è basso il segnale è stato danneggiato dal rumore e i dati non possono essere recuperati

### Controllo dell’accesso al mezzo condiviso
Per evitare collisioni (trasmissioni che si sovrappongono) è necessario controllare l’accesso al mezzo

>[!question] Perche non si può usare il CSMA/CD anche per le reti wireless?
>##### No collision detection
>Per rilevare una collisione un host deve poter trasmettere (il proprio frame) e ricevere (ascoltare il canale) conteportaneamente
>
>Poiché la potenza del segnale ricevuto è molto inferiore a quella del segnale trasmesso, sarebbe troppo costoso usare un adattatore in grado di rilevare le collisioni (i dispositivi wireless hanno un’energia limitata fornita dalla batteria che non gli consente di usare un tale dispositvo)
>
>##### Hidden terminal problem
>Un host potrebbe non accorgersi che un altro host sta trasmettendo e quindi non sarebbe in grado rilevare la collisione (ascoltando il canale)
>
>Attenuazione del segnale:
>![[Pasted image 20250517125050.png|350]]
>
>Ostacoli:
>![[Pasted image 20250517125116.png|250]]

---
## IEEE 802.11
La IEEE ha definito le specifiche per le LAN wireless, chiamate $802.11$, che coprono i livelli fisico e collegamento

Il Wi-Fi (wireless fidelity) è una LAN wireless dalla Wi-Fi Alliance, ovvero un’associazione ($300$ aziende) no profit che si occupa di promuovere la crescita delle LAN wireless

---
## Architettura
### BSS
La **BSS** (*Basic Service Set*) è una LAN wireless costituita da uno o più host wireless e da un access point

![[Pasted image 20250517125454.png]]

### ESS
La **ESS** (*Extended Service Set*) è una LAN wireless costituita da due o più BSS con infrastruttura

I BSS sono collegati tramite un sistema di distribuzione che è una rete cablata (Ethernet) o wireless. Quando i BSS sono collegati, le stazioni in visibilità comunicano direttamente, mentre le altre comunicano tramite l’AP

La ESS è un’architettura molto comune nelle reti Wi-Fi moderne, soprattutto in ambienti dove è necessario coprire aree estese con accesso continuo alla rete wireless

![[Pasted image 20250517125651.png|580]]

### Architettura generale
![[Pasted image 20250517125946.png|400]]

---
## Canali e associazione
Lo spettro $2.4 \text{ GHz}-2.485 \text{ GHz}$ è diviso in $11$ canali parzialmente sovrapposti
L’amministratore dell’AP sceglie una frequenza, ma sono possibili delle interferenze (stesso canale per AP vicini) e il numero massimo di frequenza utilizzabili da diversi AP per evitare interferenze è 3 (usando i canali 1, 6, 11)
I canali non interferiscono se separati da 4 o più canali

L’architettura IEEE 802.11 prevede che una stazione wireless si associ ad un AP per accedere a Internet

Per associare una stazione (host) ad un AP è necessario conoscere gli AP disponibili in un BSS e un protocollo di associazione

In particolare l’AP invia segnali periodici (beacon) che includono l’identificatore dell’AP (*Service Set Identifier* - **SSID**) e il suo indirizzo MAC. La stazione wireless che vuole entrare in un BSS scandisce gli $11$ canali trasmissivi alla ricerca di frame beacon (passive scanning) e alla fine della scansione, la stazione sceglie l’AP da cui ha ricevuto il beacon con la maggiore potenza di segnale e gli invia un frame con la richiesta di associazione
L’AP accetta la richiesta con un frame di risposta associazione che permetterà all’host entrante di inviare una richiesta DHCP per ottenere un indirizzo IP

>[!warning]
>Può essere prevista un’autenticazione per eseguire l’associazione

---
## Protocollo MAC 802.11
Più stazioni possono voler comunicare nello stesso momento. Sono state quindi definite due tecniche di accesso al mezzo:
- **distributed coordination function** (*DCF*) → i nodi si contendono l’accesso al canale (sistema distribuito)
- **point coordination function** (*PCF*) → non ci sta contesa e l’AP coordina l’accesso dei nodi al canale (sistema centralizzato)

D’ora in poi analizzeremo il DCF

---
## CMSA/CA
Il **CSMA/CA** permette di evitare le collisioni quando due o più nodi provano a trasmette simultaneamente (nel CSMA/CD, le collisioni venivano trovate, qui evitate)

### ACK
Poiché non è possibile effettuare collision detection come precedentemente detto, è necessario un riscontro per capire se una trasmissione è andata a buon fine. Per farlo si utilizza un **ACK**

Però il mittente non può aspettare l’ACK all’infinito per sapere se i dati sono stati ricevuti correttamente, dunque imposta un timer. Se il timer scade senza aver ricevuto l’ACK, il nodo suppone che la trasmissione sia fallita e tenta una ritrasmissione (il Wi-Fi ritrasmette fino a 7 volte)

![[Pasted image 20250517171333.png|250]]

>[!warning]
>Ci sta possibilità di collisione anche sull’ack

### Spazio interframe
Lo **spazio interframe** (*IFS*) è un intervallo di tempo che una stazione deve attendere dopo aver rilevato che il canale è libero, prima di iniziare una trasmissione. Questo meccanismo serve a evitare collisioni con altre stazioni che potrebbero aver già iniziato a trasmettere

Esistono diversi tipi di IFS, ciascuno con una priorità diversa:
- **SIFS** (Short IFS) → garantisce **alta priorità** alle trasmissioni (usato ad esempio per ACK)
- **DIFS** (Distributed IFS) → garantisce **bassa priorità**, utilizzato per le trasmissioni dati normali

>[!example] Funzionamento
>- Mittente → ascolta il canale, se libero per DIFS tempo, allora inizia a trasmettere
>- Ricevente → se il frame è ricevuto correttamente invia un ACK dopo SIFS tempo
>
>![[Pasted image 20250517171904.png|250]]

>[!hint] DIFS > SIFS
>In modo tale da dare priorità alle comunicazioni già iniziate (quindi priorità all’ack)

Se durante l’intervallo DIFS il canale diventa occupato:
- il nodo interrompe il conteggio del DIFS
- aspetta il canale torni completamente libero
- quando il canale torna libero, riavvia da zero il conteggio del DIFS completo (es. $50 \mu s$)

### Finestra di contesa
Dopo aver atteso un tempo IFS, se il canale è ancora inattivo, la stazione attende un ulteriore *tempo di contesa*

La **finestra di contesa** (*contention window*) è il lasso di tempo (backoff) per cui deve sentire il canale libero prima di trasmettere (il tempo è suddiviso in slot e ad ogni slot si esegue il sensing del canale)

In particolare si sceglie un $R$ casuale in $[0,CW-1]$, e finché $R>0$ e si ascolta il canale. Se il canale è libero per la durata dello slot $R=R-1$, altrimenti interrompe il timer e aspetta che il canale si liberi (e riavvia il timer da dove lo aveva lasciato)

![[Pasted image 20250517172552.png|550]]

Nel complesso ora si ha
- fase 1 → attesa dell’IFS (es. DIFS)
	- il nodo rileva il canale libero
	- attende un tempo fisso (es. in IEEE 802.11b $34\mu s$ per DIFS)
	- se il canale rimane libero per tutto il DIFS, si passa alla finestra di contesa
- fase 2 → finestra di contesa (backoff)
	- il nodo scegli un numero casuale in un intervallo $[0,CW-1]$, dove $CW$ è la contention window
	- ogni unità di tempo della finestra si chiama slot time (es. $20\mu s$)
	- il nodo inizia a contare all’indietro (backoff counter), solo se il canale rimane libero

>[!example] Esempio
>Se $CW=16$ l’intervallo è $[0,15]$. Supponiamo si scelga casualmente il $7$
>
>Quindi il nodo attende $7\cdot \text{slot time}=7\cdot 20\mu s=140\mu s$, solo se il canale è libero
>
>Se durante questo tempo qualcun altro inizia a trasmettere, il nodo pausa il coundown, e lo riprende da dove lo aveva interrotto una volta che il canale torna libero

### Evitare collisioni sul destinatario
Per fare in modo che le stazioni che non sono coinvolte nella comunicazione (sono nel raggio di trasmissione della destinazione ma non del mittente) sappiano quanto tempo devono astenersi dal trasmettere si utilizza l’RTS e CTS

### RTS/CTS
Il problema dell’hidden terminal non viene risolto con l’IFS e la finestra contesa. E’ dunque necessario un meccanismo di prenotazione del canale: il *request to send* (**RTS**) e *clear to send* (**CTS**)

![[Pasted image 20250517180406.png|500]]

### NAV
Quando una stazione invia un frame RTS include la durata di tempo in cui occuperà il canale per trasmettere il frame e riceve l’ACK
Questo tempo viene incluso anche nel CTS (per i vicini del destinatario)

Le stazioni influenzate da tale trasmissione avviano un timer chiamato NAV che indica quanto tempo devono attendere prima di eseguire il sensing del canale

>[!info]
>Ogni stazione, prima di ascoltare il canale, verifica il NAV

### Collisioni durante l’handshaking

>[!question] Cosa succede se avviene una collisione durante la trasmissione di RTS o CTS?
>Se il mittente non riceve CTS allora assume che c’è stata collisione e riprova dopo un tempo di backoff

### Problema della stazione esposta
Una stazione (nell’esempio C) si astiene dall’usare il canale anche se potrebbe trasmettere (C è la stazione esposta)

![[Pasted image 20250517182840.png]]

### Formato del frame

![[Pasted image 20250517182917.png]]

- **FC** (frame control) → indica il tipo di frame e alcune informazioni di controllo; una LAN wireless ha 3 categorie di frame
	-  ![[Pasted image 20250517184345.png]]
	- *00* → frame di gestione, usati per le comunicazioni iniziali tra stazioni e punti di accesso
	- *01* → frame di controllo, si usano per accedere al canale e dare discontro (si imposta il subtype in questo modo $1011$ per RTS, $1100$ per CTS, $1101$ per ACK)
	- *10* → frame di dati, vengono usati per trasportare i dati
		- frame di controllo
- **D** → durata della trasmissione, usata per impostare il NAV (impostata sia per DATA che per RTS che CTS)
- **indirizzi** → indirizzi MAC
- **SC** → informazioni sui frammenti ($\#\text{frammento e }\#\text{sequenza}$); il numero di numero di sequenza serve per distinguere frame ritrasmessi come nel livello trasporto (ACK possono andare perduti)
- **Frame body** → payload
- **FCS** → codice CRC a 32 bit

#### Frame di controllo
![[Pasted image 20250517184453.png|450]]

#### Indirizzamento
In base ai campi $\text{To DS}$ e $\text{From DS}$ del campo FC, si ha un diverso formato per i campi degli indirizzi

| Significato                    | To DS | From DS | Address 1    | Address 2   | Address 3    | Address 4 |
| ------------------------------ | ----- | ------- | ------------ | ----------- | ------------ | --------- |
| comunicazione diretta (ad-hoc) | 0     | 0       | destinazione | sorgente    | BSS ID       | N/A       |
| da AP a host                   | 0     | 1       | destinazione | AP mittente | sorgente     | N/A       |
| da host ad AP                  | 1     | 0       | AP ricevente | sorgente    | destinazione | N/A       |
| da AP ad AP                    | 1     | 1       | AP ricevente | AP mittente | destinazione | sorgente  |
In sintesi si può dire che in $\text{Address 1}$ ci sta l’indirizzo del dispositivo successivo a cui viene trasmesso il frame, mentre in $\text{Address 2}$ l’indirizzo del dispositivo che il frame ha lasciato

![[Pasted image 20250517185352.png|400]]
![[Pasted image 20250517185422.png|400]]
![[Pasted image 20250517185438.png|400]]
![[Pasted image 20250517185503.png|400]]

---
## Mobilità all’interno della stessa sottorete IP

---
Created: 2025-03-13
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Protocollo|Protocollo]]
- [[#Organizzazione a più livelli|Organizzazione a più livelli]]
	- [[#Organizzazione a più livelli#Principi della strutturazione a livelli|Principi della strutturazione a livelli]]
	- [[#Organizzazione a più livelli#Collegamento logico fra i livelli|Collegamento logico fra i livelli]]
- [[#Lo stack protocollare TCP/IP|Lo stack protocollare TCP/IP]]
	- [[#Lo stack protocollare TCP/IP#Pila di protocolli TCP/IP|Pila di protocolli TCP/IP]]
		- [[#Pila di protocolli TCP/IP#Applicazione|Applicazione]]
		- [[#Pila di protocolli TCP/IP#Trasporto|Trasporto]]
		- [[#Pila di protocolli TCP/IP#Rete|Rete]]
		- [[#Pila di protocolli TCP/IP#Link|Link]]
		- [[#Pila di protocolli TCP/IP#Fisico|Fisico]]
	- [[#Lo stack protocollare TCP/IP#Comunicazione in una internet|Comunicazione in una internet]]
- [[#Gerarchia dei protocolli|Gerarchia dei protocolli]]
- [[#Servizi e protocolli|Servizi e protocolli]]
	- [[#Servizi e protocolli#Architettura di rete|Architettura di rete]]
- [[#Incapsulamento e decapsulamento|Incapsulamento e decapsulamento]]
	- [[#Incapsulamento e decapsulamento#Multiplexing e demultiplexing|Multiplexing e demultiplexing]]
- [[#Indirizzamento nel modello TCP/IP|Indirizzamento nel modello TCP/IP]]
- [[#Layering|Layering]]
	- [[#Layering#Vantaggi|Vantaggi]]
	- [[#Layering#Svantaggi|Svantaggi]]
- [[#Modello OSI|Modello OSI]]
	- [[#Modello OSI#Confronto tra OSI e TCP/IP|Confronto tra OSI e TCP/IP]]
	- [[#Modello OSI#Insuccesso del modello OSI|Insuccesso del modello OSI]]
- [[#Gli standard|Gli standard]]
	- [[#Gli standard#Livello di maturità|Livello di maturità]]
- [[#Livello applicazione|Livello applicazione]]
	- [[#Livello applicazione#Offerta di servizi|Offerta di servizi]]
		- [[#Offerta di servizi#Protocolli standard|Protocolli standard]]
		- [[#Offerta di servizi#Protocolli non standard|Protocolli non standard]]
	- [[#Livello applicazione#Architettura di un’applicazione|Architettura di un’applicazione]]
		- [[#Architettura di un’applicazione#Paradigma client-server|Paradigma client-server]]
---
## Introduction
Si è fornita una panoramica della struttura e delle prestazioni di Internet, che è costituita da numerose reti di varie dimensioni interconnesse tramite opportuni dispositivi di comunicazione. Tuttavia per poter comunicare non è sufficiente assicurare questi collegamenti, ma è necessario utilizzare sia dell’hardware che del software (**hardware e software devono essere coordinati**)

>[!example]- Esempio di comunicazione
>![[Pasted image 20250313201724.png]]
>Dure interlocutori rispettano un protocollo di conversazione (interazione).
>- Si inizia con un saluto
>- Si adotta un linguaggio appropriato al livello di conoscenza
>- Si tace mentre l’altro parla
>- La conversazione si sviluppa come dialogo piuttosto che un monologo
>- Si termina con un saluto

---
## Protocollo
Un protocollo **definisce le regole** che il mittente e il destinatario, così come tutti i sistemi coinvolti, devono rispettare per essere in grado di comunicare.

In situazioni particolarmente semplici potrebbe essere sufficiente un solo protocollo, in situazioni più complesse potrebbe essere opportuno suddividere i compiti fra più **livelli** (*layer*), nel qual caso è richiesto un protocollo per ciascun livello (si parla di *layering di protolli*)

---
## Organizzazione a più livelli

>[!example]
>Anna viene trasferita e le due amiche continuano a sentirsi via posta. Poiché hanno in mente un progetto innovativo vogliono rendere sicura la conversazione mediante un meccanismo di crittografia. Il mittente di una lettera la cripta per renderla incomprensibile a un eventuale intruso, il destinatario la decripta per recuperare il messaggio originale
>
>![[Pasted image 20250313215922.png|500]]
>- Si ipotizza che le due amiche abbiano tre macchine ciascuna per portare a termine i compiti di ciascun livello
>- Supponiamo che Maria invii la prima lettera
>- Maria comunica con la macchina al terzo livello come se fosse Anna e la potesse ascoltare
>
>La strutturazione dei protocolli in livelli consente di **suddividere un compito complesso in compiti più semplici**.
>
>Si potrebbe usare una sola macchina ma cosa accadrebbe se le due amiche decidessero di cambiare la tecnica di crittografia? Usando le 3 macchine dell’esempio verrebbe sostituita solo quella intermedia (**modularizzazione**, indipendenza dei livelli)

### Principi della strutturazione a livelli
Quando è richiesta una comunicazione **bidirezionale**, ciascun livello deve essere capace di effettuare i due compiti opposti, uno per ciascuna direzione (es. crittografare, decrittografare).

Gli oggetti di input/output sotto ciascun livello di entrambi i lati devono essere identici

### Collegamento logico fra i livelli
I livelli logicamente sono direttamente collegati, ovvero il protocollo implementato a ciascun livello specifica una comunicazione diretta fra i pari livelli delle due parti

---
## Lo stack protocollare TCP/IP
**TCP/IP** è una famiglia di protocolli attualmente utilizzata in Internet. Si tratta di una gerarchia di protocolli costituita da moduli interagenti, ciascuno dei quali fornisce funzionalità specifiche.
Il termine **gerarchia** significa che ciascun protocollo di livello superiore è supportato dai servizi forniti dai protocolli di livello inferiore.

Definita in origine in termini di quattro livelli software soprastanti a un livello hardware, la pila TCP/IP è oggi intesa come composta di **cinque livelli**.
![[Pasted image 20250313221458.png|500]]

### Pila di protocolli TCP/IP
#### Applicazione
L’applicazione è la sede delle applicazioni di rete. I protocolli sono:
- HTTP
- SMTP
- FTP
- DNS

Qui i pacchetti sono denominati **messaggi**. E’ solo software

#### Trasporto
Il trasporto consiste nel trasferimento dei messaggi a livello di applicazione tra il modulo client e server di un’applicazione. I protocolli sono:
- TCP
- UDP

Qui i pacchetti sono denominati **segmenti**. E’ solo software

#### Rete
La rete è l’instradamento dei segmenti dall’origine alla destinazione (*end-to-end*). I protocolli sono:
- IP
- protocolli di instradamento

Qui i pacchetti sono denominati **datagrammi**. E’ parzialmente software e parzialmente hardware

#### Link
Un link trasmette datagrammi da un nodo a quello successivo sul percorso (*hop-to-hop*). I protocolli sono:
- Ethernet
- Wi-Fi
- PPP

Seppur lungo un percorso sorgente-destinazione un datagramma può essere gestito da protocolli diversi.
Qui i pacchetti sono denominati **frame**. E’ solo hardware

#### Fisico
Si occupa del trasferimento dei singoli bit. E’ solo hardware

### Comunicazione in una internet
I sistemi intermedi richiedono **solo alcuni livelli**, e grazie al layering tali sistemi implementano solo i livelli necessari, riducendo la complessità. Dunque lo stack protocollare si trova su qualsiasi dispositivo che si trova in rete

![[Pasted image 20250313222154.png]]
![[Pasted image 20250313222212.png|500]]

---
## Gerarchia dei protocolli
La rete è organizzata come pila di **strati** (layer) o **livelli**, costruiti l’uno sull’altro. Lo scopo di ogni strato è quello di **offrire servizi** agli strati di livello superiore, nascondendo i dettagli di implementazione

Lo strato $N$ di un computer è in comunicazione con lo strato $N$ di un altro computer. Le regole e le convenzioni usate in questa comunicazione sono globalmente note come **protocolli** dello strato $N$

Le entità che formano gli strati sono chiamati **peer**, i quali comunicano usando il protocollo

>[!warning]
>I dati non sono trasferiti direttamente dallo strato $N$ di un computer allo strato $N$ di un altro computer

---
## Servizi e protocolli
Servizi e protocolli sono concetti ben distinti.
Un **servizio** è un insieme di primitive che uno strato offre a quello superiore. Definisce quindi **quali operazioni lo strato è in grado di offrire**, ma non dice nulla di come queste operazioni sono implementate.
Un **protocollo** è un insieme di regole che controllano il formato e il significato dei pacchetti, o messaggi scambiati tra le entità pari all’interno di uno strato
![[Pasted image 20250313223021.png|center|400]]

### Architettura di rete
Ogni strato passa dati e informazioni di controllo allo stato immediatamente sottostante fino a raggiungere quello più in basso

![[Pasted image 20250313223446.png]]

![[Pasted image 20250313223520.png|600]]

---
## Incapsulamento e decapsulamento
La sorgente effettua l’incapsulamento (prende il pacchetto dal livello superiore, lo considera come carico dati o payload e aggiunge un header)
- **Messaggio** → nessuna intestazione
- **Segmento** (o datagramma utente) → header trasporto + messaggio
- **Datagramma** → header rete + segmento
- **Frame** → header collegamento + segmento
Il destinatario invece effettua il decapsulamento. Il router effettua sia incapsulamento che decapsulamento perché collegato a due link

![[Pasted image 20250313223920.png]]

### Multiplexing e demultiplexing
Dato che lo stack protocollare TCP/IP prevede più protocolli nello stesso livello, è necessario eseguire il multiplexing alla sorgente e il demultiplexing alla destinazione.

**Multiplexing**: un protocollo può incapsulare (uno alla volta) i pacchetti ottenuti da più protocolli del livello superiore. **Demultiplexing**: un protocollo può decapsulare e consegnare i pacchetti a più protocolli del livello superiore

E’ necessario un campo nel proprio header per identificare a quale protocollo appartengono i pacchetti incapsulati

![[Pasted image 20250313224323.png]]

![[Pasted image 20250313224450.png]]

---
## Indirizzamento nel modello TCP/IP
Poiché il modello TCP/IP prevede una comunicazione logica tra coppie di livelli è necessario avere un indirizzo sorgente e un indirizzo destinazione per ogni livello

![[Pasted image 20250313224600.png]]

---
## Layering
### Vantaggi
**Modularità**, che comporta:
- Semplicità di design
- Possibilità di modificare un modulo in modo trasparente se le interfacce con gli altri livelli rimangono le stesse
- Possibilità per ciascun costruttore di adottare la propria implementazione di un livello purché requisiti su interfacce soddisfatti

![[Pasted image 20250313224751.png|500]]

### Svantaggi
A volte necessario scambio di informazioni tra livelli non adiacenti (es. per ottimizzare app
funzionante su wireless) non rispettando principio della stratificazione

---
## Modello OSI
L’**ISO** (*International Organization for Standardization*), organizzazione dedicata alla definizione di standard universalmente accettati, ha definito il modello **OSI** (*Open System Interconnection*) come modello alternativo al TCP/IP

![[Pasted image 20250313225031.png|400]]

### Confronto tra OSI e TCP/IP
![[Pasted image 20250313225104.png|550]]

### Insuccesso del modello OSI
L’OSI venne pubblicato quando il **TCP/IP era già ampiamente diffuso** e gli erano state dedicate parecchie risorse, dunque un’eventuale sostituzione avrebbe comportato un costo notevole, tanto che alcuni livelli, come presentazione e sessione non sono mai stati completamente specificati (software corrispondente mai stato completamente sviluppato).
Inoltre non riuscirono a dimostrare delle prestazioni tali da convincere le autorità di Internet a sostituire il TCP/IP

---
## Gli standard
Nello studio di Internet e dei suoi protocolli si incontrano spesso riferimenti a standard o entità amministrative
Uno standard Internet è una specifica che è stata **rigorosamente esaminata e controllata**, utile e accettata da chi utilizza la rete Internet.

Si tratta di un insieme di regole formalizzate che devono **necessariamente essere seguite**.
Ma prima che esse siano approvate, esiste una **procedura rigorosa** attraverso la quale una
specifica diviene uno standard Internet. Il primo stadio di una specifica è quello di bozza Internet (*Internet draft*).

### Livello di maturità
- Proposta di standard
- Draft standard
- Standard Internet
- Livello storico
- Livello sperimentale
- Livello informativo

![[Pasted image 20250313225612.png|400]]

---
## Livello applicazione
Il livello applicazione fornisce **servizi all’utente**.
La comunicazione è fornita per mezzo di una **connessione logica**: questo significa
che i livelli applicazione nei due lati della comunicazione agiscono **come se esistesse un collegamento diretto** attraverso il quale poter inviare e ricevere messaggi, nonostante la comunicazione reale avviene attraverso più livelli, più dispositivi e vari canali fisici

![[Pasted image 20250313230039.png|500]]

### Offerta di servizi
Internet è stata progettata per fornire servizi agli utenti. Dato che il livello applicazione è l’unico che fornisce servizi agli utenti di Internet, la sua flessibilità consente di **aggiungere nuovi protocolli con estrema facilità**, così come si è verificato nella storia di Internet e sta tuttora avvenendo.

Un protocollo che viene aggiunto a un dato livello deve essere progettato in modo da usare i servizi del livello inferiore, ma aggiungere o eliminare protocolli dal livello applicazione è relativamente facile perché non comporta modifiche agli altri livelli
#### Protocolli standard
Esistono diversi protocolli di livello applicazione che sono standardizzati e documentati dagli enti responsabili della gestione di Internet.
Ogni protocollo standard è costituito da una coppia di programmi che interagiscono con l’utente e con il livello di trasporto per fornire uno specifico servizio. Un esempio è un’applicazione Web specificata dal protocollo di comunicazione HTTP
#### Protocolli non standard
E’ possibile creare un’applicazione non standard scrivendo due programmi che forniscono servizi agli utenti, facendo uso dei servizi di trasporto
Non è necessario chiedere autorizzazioni: un’azienda può sviluppare il proprio protocollo di livello applicazione per far comunicare i propri uffici sparsi nel mondo

### Architettura di un’applicazione
I due programmi applicativi devono essere entrambi in grado di richiedere e offrire servizi,
oppure ciascuno deve occuparsi di uno dei due compiti?

Esistono tre differenti paradigmi:
- Client-server
- Peer-to-peer (P2P)
- Architetture ibride (client-server e P2P)
#### Paradigma client-server
![[Pasted image 20250313230815.png|600]]
Il **client** è il richiedente del del servizio e rimane in esecuzione solo quando il servizio è necessario. Numerosi client che richiedono i servizi
Il **server** invece fornisce i servizi e rimane sempre in esecuzione in attesa di richieste client. Numero limitato di processi server pronti a offrire uno specifico servizio

>[!info] Il ruolo delle due entità è totalmente differente: non è possibile eseguire un client come programma server e viceversa

Svantaggi:
- Carico di comunicazione risulta concentrato sul server (che deve essere molto potente)
- Server farm per creare un potente server virtuale
- Costi di gestione per offrire servizio
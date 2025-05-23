---
Created: 2025-05-07
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Index
- [[#Struttura di internet|Struttura di internet]]
	- [[#Struttura di internet#Impossibilità di usare un singolo protocollo di routing|Impossibilità di usare un singolo protocollo di routing]]
- [[#Instradamento gerarchico|Instradamento gerarchico]]
- [[#Sistemi autonomi|Sistemi autonomi]]
	- [[#Sistemi autonomi#Sistemi autonomi interconnessi|Sistemi autonomi interconnessi]]
		- [[#Sistemi autonomi interconnessi#Routing intra-dominio|Routing intra-dominio]]
		- [[#Sistemi autonomi interconnessi#Routing inter-dominio|Routing inter-dominio]]
- [[#Border Gateway Protocol|Border Gateway Protocol]]
- [[#Path-vector routing|Path-vector routing]]
	- [[#Path-vector routing#Aggiornamento dei path vector|Aggiornamento dei path vector]]
	- [[#Path-vector routing#Algoritmo path vector|Algoritmo path vector]]
- [[#eBGP e iBGP|eBGP e iBGP]]
	- [[#eBGP e iBGP#eBGP|eBGP]]
	- [[#eBGP e iBGP#iBGP|iBGP]]
		- [[#iBGP#Scambio di messaggi|Scambio di messaggi]]
	- [[#eBGP e iBGP#Tabelle di percorso|Tabelle di percorso]]
	- [[#eBGP e iBGP#Tabelle di routing|Tabelle di routing]]
- [[#Attributi del percorso e rotte BGP|Attributi del percorso e rotte BGP]]
- [[#Selezione dei percorsi BGP|Selezione dei percorsi BGP]]
	- [[#Selezione dei percorsi BGP#Advertising ristretto|Advertising ristretto]]
- [[#Messaggi BGP|Messaggi BGP]]
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
RIP e OSPF vengono utilizzati per determinare i percorsi ottimali per le coppie origine-destinazione interne a un sistema autonomo
Il BGP (Border Gateway Protocol), proprietà di CISCO, viene usato per determinate i percorsi per le coppie origine-destinazione che interessano più sistemi autonomi e che rappresenta l’attuale standard de facto

Il BGP è un protocollo **path vector** (distance vector con percosi) e che mette a disposizione di ciascun AS un modo per:
1. ottenere informazioni sulle raggiungibilità delle sottoreti da parte di AS confinanti
2. propagare le informazioni di raggiungibilità a tutti i router interni ad un AS
3. determinare percorsi “buoni” (non necessariamente basati sull’ottimizzazione) verso le sottoreti sulla base delle informazioni di raggiungibilità e delle politiche dell’AS

>[!hint]
>BPG consente a ciascuna sottorete di comunicare la propria esistenza al resto di internet

---
## Path-vector routing
Sia LS che DV si basano sul costo minimo, tuttavia ci sono casi in cui il costo minimo non è l’obiettivo prioritario

>[!example]
>Un mittente non vuole che i suoi pacchetti passino attraverso determinati router

Il routing a costo minimo non consente di applicare questo tipo di politiche nella scelta del percorso, mentre con il path-vector routing la sorgente può controllare il percorso così da minimizzare il numero di hop ed evitare alcuni nodi

Risulta di fatto simile al distance vector ma avengono inviati percorsi invece che solo destinazioni. Ogni nodo, quando riceve un path vector da un vicino, aggiorna il suo path vector applicando la sua politica invece del costo minimo

Inizializzazione:
![[Pasted image 20250508112752.png]]

### Aggiornamento dei path vector

>[!question] Cosa succede quando $C$ riceve una copia del vettore di $B$?
>![[Pasted image 20250508112914.png|360]]

>[!question] Cosa succede quando $C$ riceve una copia del vettore di $D$?
>Nessun cambiamento
>![[Pasted image 20250508113009.png|360]]

### Algoritmo path vector

```
Path_Vector_Routing() {
	// Inizializzazione
	for (y=1 to N) {
		if (y è me_stesso)
			Path[y] = me_stesso
		else if (y è un vicino)
			Path[y] = me_stesso+il_nodo_vicino
		else
			Path[y] = vuoto
	}
	Spedisci il vettore {Path[1], Path[2], ..., Path[y]} a tutti i vicini
	
	// Aggiornamento
	repeat (sempre) {
		wait (un vettore Path_w da un vicino w)
		for (y=1 to N) {
			if (Path_w comprende me_stesso)
				scarta il percorso
			else
				Path[y] = il_migliore_tra{Path[y], (me_stesso+Path_w[y])}
		}
		if (c'è un cambiamento nel vettore)
			Spedicsci il vettore {Path[1], Path[2], ..., Path[y]} a tutti i vicini
	}
}
```

---
## eBGP e iBGP
Per permettere ad ogni router di instradare correttamente i pacchetti, qualsiasi sia la destinazione, è necessario installare su tutti i **router di confine** (*border router*) dell’AS una variante del BGP chiamata **external BGP** (*eBGP*)

Tutti i router (non solo quelli di confine) dovranno invece usare la seconda variante del BGP, chiamata **internal BGP** (*iBGP*)

Dunque i router di confine devono eseguire tre protocolli di routing (intra-dominio, eBGP, iBGP), mentre tutti gli altri ne eseguono due (intra-dominio, iBGP)

![[Pasted image 20250508113814.png]]

>[!info] Fondamenti di BGP
>- coppie di router si scambiano informazioni di instradamento su connessioni TCP usando la porta $179$
>- i router ai capi di una connessione TCP sono chiamati **peer BGP**, e la connessione TCP con tutti i messaggi che vi vengono inviati è detta **sessione BGP**
>- notiamo che le linee di sessione BGP non sempre corrispondono ai collegamenti fisici

### eBGP
Due router di confine che si trovano in due diversi AS formano una coppia di peer BGP e si scambiano messaggi

![[Pasted image 20250508114154.png]]

I messaggi scambiati durante le sessioni eBGP servono per indicare ad alcuni router come instradare i pacchetti destinati ad alcune reti, ma le informazioni di raggiungibilità non sono complete.
Problemi da risolvere:
1. i router di confine sanno instradare pacchetti solo ad AS vicini
2. nessuno dei router di confine (interno agli AS) sa come instradare un pacchetto destinato alle reti che si trovano in altri AS

Come soluzione ci sta l’**iBGP**

### iBGP
L’iBGP crea una sessione tra ogni possibile coppia di router all’interno di un AS; quindi nonostante non tutti i nodi hanno messaggi da inviare, tutti li ricevono

![[Pasted image 20250508114518.png]]

#### Scambio di messaggi
Il processo di aggiornamento non termina dopo il primo scambio di messaggi, ma continua finché non ci sono più aggiornamenti

>[!example]
>$R1$ dopo che ha inviato il messaggio di aggiornamento a $R2$, combina le informazioni circa la raggiungibilità di $AS3$ con quelle che già conosceva relativamente a $AS1$ e invia un nuovo messaggio di aggiornamento a $R5$ (che quindi sa come raggiungere $AS1$ e $AS3$)

Le informazioni ottenute da eBGP e iBGP vengono combinate per creare le tabelle dei percorsi

### Tabelle di percorso
![[Pasted image 20250508115109.png]]
![[Pasted image 20250508115219.png]]

### Tabelle di routing
Le tabelle di percorso ottenute da BGP non vengono usate di per sé per l’instradamento dei pacchetti bensì inserite nelle tabelle di routing intra-dominio (generate da RIP o OSPF)

Nel caso di stub, l’unico router di confine dell’area aggiunge una regola di default alla fine della sua tabella di routing e definisce come prossimo router quello che si trova dall’altro lato della connessione
Nel caso di AS di transito, il contenuto della tabella di percorso deve essere inserito nella tabella di routing, ma bisogna impostare il costo (RIP e OSPF usano metriche differenti) pari a quello per raggiungere il primo AS nel percorso

>[!example] Tabelle di inoltro dopo l’aggiunta delle informazioni BGP
>Nel caso di stub l’unico router di confine dell’area aggiunge una regola di default alal fine della sua tabella di routing e definisce come prossimo router quello che si trova dall’altro lato della connessione eBGP
>
>![[Pasted image 20250508115719.png]]
>
>Nel caso di AS di transito, il contenuto della tabella di percorso deve essere inserito nella tabella di routing ma bisogna impostare il costo
>
>![[Pasted image 20250508115835.png]]

---
## Attributi del percorso e rotte BGP
Quando un router annuncia una rotta per un prefisso (di rete) per una connessione BGP, include anche un certo numero di **attributi BGP** ($\text{prefisso}+\text{attributi}=\text{"rotta"}$)

Due dei più importanti attributi sono:
- **AS-PATH** → serve per selezionare i percorsi (ed evitare cicli)
	- elenca i sistemi autonomi attraverso i quali è passato l’annuncio del prefisso (e quindi gli hop intermedi della rotta): ogni sistema autonomo non ha un identificativo univoco
- **NEXT-HOP** → indirizzo IP dell’interfaccia su cui viene inviato il pacchetto (un router ha più indirizzi IP, uno per ogni inferfaccia)

Quando un router gateway riceve un annuncio di rotta, utilizza le proprie **politiche d’importazione** per decidere se accettare o filtrare la rotta. Infatti il sistema autonomo può non voler inviare traffico su uno degli AS presenti nel AS-PATH (il router conosce la rotta migliore)

---
## Selezione dei percorsi BGP
Un router può ricavare più di una rotta verso una destinazione (percorsi multipli), e deve quindi sceglierne una

Regole di eliminazione:
1. alle rotte viene assegnato un valore di **preferenza locale**. Si selezionano quindi le rotta con i più alti valori di preferenza locale (riflette la politica imposta dall’amministratore)
2. si seleziona la rotta con valore AS-PATH più breve
3. si seleziona quella con il router di NEXT-HOP a costo minore (*hot-potato routing*)
4. se rimane ancora più di una rotta, il router si basa sugli identificatori BGP

### Advertising ristretto

>[!example]
>Gli ISP vogliono istradare solo il traffico delle loro customer network (non vogliono instradare il traffico di transito per le altre reti)
>
>![[Pasted image 20250508120813.png]]
>
>- $A$ annuncia il percorso $Aw$ a $B$ e a $C$
>- $B$ sceglie di non annunciare $BAw$ a $C$
>	- $B$ non ha vantaggio a instradare $CBAw$, poiché nessuno tra $C$, $A$, $w$ sono clienti di $B$
>	- $C$ non scopre il percorso $CBAw$
>- $C$ instraderà solo $CAw$ (senza usare $B$) per raggiungere $w$

>[!example]
>![[Pasted image 20250508120813.png]]
>
>Analizziamo questa rete rete:
>- $A$, $B$, $C$ → provider networks
>- $x$, $w$, $y$ → customer
>- $x$ → dual-homed (collegata a due reti)
>- policy to enforce → $x$ non vuole instradare da $B$ a $C$ via $x$, allora $x$ non annuncia a $B$ la rotta verso $C$

---
## Messaggi BGP
I messaggi BGP vengono scambiati attraverso TCP

Messaggi BGP:
- `OPEN` → apre la connessione TCP e autentica il mittente
- `UPDATE` → annuncia il nuovo percorso (o cancella quello vecchio)
- `KEEPALIVE` → mantiene la connessione attiva in mancanza di `UPDATE`
- `NOTIFICATION` → riporta gli errori del precedente messaggio; usato anche per chiudere il collegamento

>[!question] Perché i protocolli d’instradamento inter-AS sono diversi da quelli intra-AS?
>Politiche:
>- inter-AS → il controllo amministrativo desidera avere il controllo su come il traffico viene instradato e su chi instrada attraverso le sue reti
>- intra-AS → unico controllo amministrativo, e di conseguenza e questioni di politica hanno un ruolo molto meno importante nello scegliere le rotte interne al sistema
>
>Prestazioni:
>- intra-AS → orientato alle prestazioni
>- inter-AS → le politiche possono prevalere sulle prestazioni


---
Created: 2025-04-04
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#API|API]]
- [[#Comunicazione tra processi|Comunicazione tra processi]]
- [[#Indirizzamento dei processi|Indirizzamento dei processi]]
	- [[#Indirizzamento dei processi#Come viene recapitato un pacchetto all’applicazione|Come viene recapitato un pacchetto all’applicazione]]
- [[#Numeri di porta|Numeri di porta]]
- [[#Individuare i socket address|Individuare i socket address]]
	- [[#Individuare i socket address#Individuare i socket address lato client|Individuare i socket address lato client]]
	- [[#Individuare i socket address#Individuare i socket address lato server|Individuare i socket address lato server]]
- [[#Utilizzo dei servizi di livello trasporto|Utilizzo dei servizi di livello trasporto]]
	- [[#Utilizzo dei servizi di livello trasporto#Quale servizio richiede l’applicazione?|Quale servizio richiede l’applicazione?]]
		- [[#Quale servizio richiede l’applicazione?#Perdita di dati|Perdita di dati]]
		- [[#Quale servizio richiede l’applicazione?#Temporizzazione|Temporizzazione]]
		- [[#Quale servizio richiede l’applicazione?#Throughput|Throughput]]
		- [[#Quale servizio richiede l’applicazione?#Sicurezza|Sicurezza]]
- [[#Programmazione con socket|Programmazione con socket]]
- [[#Terminologia|Terminologia]]
- [[#Programmazione socket con TCP|Programmazione socket con TCP]]
	- [[#Programmazione socket con TCP#Package $\verb|java.net|$|Package $\verb|java.net|$]]
	- [[#Programmazione socket con TCP#Interazione client/server|Interazione client/server]]
	- [[#Programmazione socket con TCP#Client Java (TCP)|Client Java (TCP)]]
	- [[#Programmazione socket con TCP#Server Java (TCP)|Server Java (TCP)]]
- [[#Programmazione socket con UDP|Programmazione socket con UDP]]
	- [[#Programmazione socket con UDP#Interazione client/server|Interazione client/server]]
	- [[#Programmazione socket con UDP#Client Java (UDP)|Client Java (UDP)]]
	- [[#Programmazione socket con UDP#Server Java (UDP)|Server Java (UDP)]]
---
## Introduction
Nel paradigma client/server la comunicazione a livello applicazione avviene tra due programmi applicativi in esecuzione chiamati processi: un client e un server
- un client è un programma in esecuzione che inizia la comunicazione inviando una richiesta
- un server è un altro programma applicativo che attende le richieste dai client

---
## API
Un linguaggio di programmazione prevede un insieme di istruzioni matematiche (un insieme di istruzioni per la manipolazione delle stringhe etc.)
Se si vuole sviluppare un programma capace di comunicare con un altro programma, è necessario un nuovi insieme di istruzioni per chiedere ai primi quattro livelli dello stack TCP/IP di aprire la connessione, inviare/ricevere dati e chiudere la connessione

Un insieme di istruzioni di questo tipo viene chiamato **API** (*Application Programming Interface*)

![[Pasted image 20250404115437.png|550]]

---
## Comunicazione tra processi
La comunicazione tra i processi avviene tramite il **socket**

Il **socket** appare come un terminale o un file ma non è un’entità fisica. E’ infatti una struttura dati creata ed utilizzata dal programma applicativo per comunicare tra un processo client e un processo server (equivale a comunicare tra due socket create nei due socket create nei due lati di comunicazione)

![[Pasted image 20250323170213.png]]

Un socket address è composto da indirizzo IP e numero di porta
![[Pasted image 20250323170357.png|500]]

---
## Indirizzamento dei processi
Affinché un processo su un host invii un messaggio a un processo su un altro host, il mittente deve identificare il processo destinatario
Un host ha un indirizzo IP univoco a $32\text{ bit}$ ma non è sufficiente questo per identificare anche il processo (sullo stesso host possono essere in esecuzione più processi) ma è necessario anche il **numero di porta** associato al processo

### Come viene recapitato un pacchetto all’applicazione
![[Pasted image 20250404120414.png|400]]

---
## Numeri di porta
I numeri di porta sono contenuti in $16 \text{ bit}$ ($0-65535$). Svariate porte sono usate da server noti (FTP 20, TELNET 23, SMTP 25, HTTP 80, POP3 110 etc.)

L’assegnamento delle porte segue queste regole:
- $0$ → non usata
- $1-255$ → riservate per processi noti
- $256-1023$ → riservate per altri processi
- $1024-65535$ → dedicate alle app utente

---
## Individuare i socket address
L’interazione tra client e server è **bidirezionale**. E’ necessaria quindi una coppia di indirizzi socket: **locale** (mittente) e **remoto** (destinatario); l’indirizzo locale in una direzione e l’indirizzo remoto nell’altra

### Individuare i socket address lato client
Il client ha bisogno di un socket address locale (client) e uno remoto (server) per comunicare

Il socket address **locale** viene **fornito dal sistema operativo**, infatti il SO conosce l’indirzzo IP del computer su cui il client è in esecuzione e il numero di porta è assegnato temporaneamente dal sistema operativo (numero di porta effimero, non viene utilizzato da latri processi)

Per quanto riguarda il socket address **remoto** il numero di porta è noto in base all’applicazione, mentre l’indirizzo IP è fornito dal DNS (oppure porta e indirizzo noti al programmatore quando si vuole verificare il corretto funzionamento di un’applicazione)

### Individuare i socket address lato server
Il server ha bisogno di un socket address locale (client) e uno remoto (server) per comunicare

Il socket address **locale** viene fornito dal sistema operativo, infatti il SO conosce l’indirzzo IP del computer su cui il server è in esecuzione e il numero di porta è **assegnato dal progettista** (numero well known o scelto)

Il socket address remoto è il socket address locale del client che si connette e poiché numerosi client possono connettersi, il server non può conoscere a priori tutti i socket address, ma li trova all’interno del pacchetto di richiesta

>[!warning]
>Il socket address locale di un server non cambia (è fissato e rimane invariato), mentre il socket address remoto varia ad ogni interazione con client diversi (anche con stesso client su connessioni diverse). Infatti se mi connetto da due browser allo stesso server cambierà il socket (hanno porte diverse); quindi dal server si riceveranno due risposte su due porte diverse

---
## Utilizzo dei servizi di livello trasporto
Una coppia di processi fornisce servizi agli utenti Internet, siano questi persone o applicazioni.
La coppia di processi, tuttavia, deve utilizzare i servizi offerti dal livello trasporto per la comunicazione, poiché non vi è comunicazione fisica a livello applicazione

Nel livello trasporto della pila di protocolli TCP/IP sono previsti due protocolli principali:
- protocollo UDP
- protocollo TCP

### Quale servizio richiede l’applicazione?
#### Perdita di dati
Alcune applicazione (es. audio) possono tollerare qualche perdita, mentre altre applicazioni (es. trasferimento dati) richiedono un trasferimento dati affidabile al 100%
#### Temporizzazione
Alcune applicazioni (es. giochi, Internet) per essere “realistiche” richiedono piccoli ritardi, mentre altre applicazioni (es. posta elettronica) non hanno particolari requisiti di temporizzazione
#### Throughput
Alcune applicazioni (es. multimediali) per essere efficaci richiedono un’ampiezza di banda minima, mentre altre applicazioni (“le applicazioni elastiche”) utilizzano l’ampiezza di banda che si rende disponibile
#### Sicurezza
Cifratura, integrità dei dati, …

---
## Programmazione con socket
La **socket API** è stato introdotta in BDS4.1 UNIX nel 1981. Questa viene esplicitamente creata, usata, distribuita dalle applicazioni secondo il paradigma client/server

Si hanno due tipi di servizio di trasporto tramite socket API:
- datagramma inaffidabile
- affidabile, orientata ai byte

![[Pasted image 20250404122435.png|550]]

**Pre-requisiti per contattare il server**:
- il processo server deve essere in esecuzione
- il server deve aver creato un socket che dà il benvenuto al contatto con il client

**Il client contatta il server**:
- creando un socket TCP
- specificando l’indirizzo IP, il numero di porta del processo server
- una volta fatto ciò il client TCP stabilisce una connessione con il server TCP

![[Pasted image 20250404122919.png|400]]

Quando viene contattato dal client, il server TCP crea un nuovo socket per il processo server per comunicare con il client, così fa poter comunicare con più client utilizzando i numeri di porta di origine per distinguerli

---
## Terminologia
- **Flusso** (*stream*) → una sequenza di caratteri che fluisce verso/da un processo
- **Flusso d’ingresso** (*input stream*) → collegato a un’origine di input per il processo ad esempio la tastiera o la socket
- **Flusso di uscita** (*output stream*) → collegato a un’uscita per il processo, ad esempio ul monitor o la socket

---
## Programmazione socket con TCP
Un’esempio di applicazione client-server è così strutturata:
1. Il client legge un riga dall’input standard (flusso `inFromUser`) e la invia al server tramite la socket (flusso `outToServer`)
2. Il server legge la riga dalla socket
3. Il server converte la riga in lettere maiuscole e la invia al client
4. Il client legge nella sua socket la riga modificata e la visualizza (flusso `inFromServer`)

![[Pasted image 20250404191853.png|240]]
### Package $\verb|java.net|$
Il package `java.net` fornisce interfacce e classi per l’implementazione di applicazioni di rete:
- le classi `Socket` e `ServerSocket` per le connessioni TCP
- la classe `DatagramSocket` per le connessioni UDP
- la classe `URL` per le connessioni HTTP
- la classe `InetAddress` per rappresentare gli indirizzi Internet
- la classe `URLConnection` per rappresentare le connessioni ad un URL

### Interazione client/server
![[Pasted image 20250404191252.png|center|500]]

### Client Java (TCP)

```java
import java.io.*;
import java.net.*;

class TCPClient {
	public static void main(String argv[]) throws Exception {
		String sentence;
		String modifiedSencence;
		
		// crea un flusso d'ingresso
		BufferedReader inFromUser = new BufferedReader(new InputStreamReader(System.in));
		// crea un socket client, connesso al server
		Socket clientSocket = new Socket("hostname", 6789);
		// crea un flusso di uscita collegato al socket
		DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
		// crea un flusso di d'ingrasso collegato alla socket
		BufferedReader inFromServer = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
		
		System.out.print("Inserisci una frase: ");
		sentence = inFromUser.readLine();
		
		// invia la frase inserita dall'utente al server
		outToServer.writeBytes(sentence + '\n');
		// legge la risposta dal server
		modifiedSentence = inFromServer.readLine();
		
		System.out.println("FROM SERVER: " + modifiedSentence);
		
		// chiude socket e connessione con server
		clientSocket.close();
	}
}
```

### Server Java (TCP)

```java
import java.io.*;
import java.net.*;

class TCPServer {
	public static void main(String argv[]) throws Exception {
		String clientSentence;
		String capitalizedSentence;
		
		// crea un socket di benvenuto sulla porta 6789
		ServerSocket welcomeSocket = new ServerSocket(6789);
		while(true) {
			// attende, sul socket di benvenuto, un contatto dal client
			Socket connectionSocket = welcomeSocket.accept();
			// crea un flusso d'ingresso collegato al socket
			BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
			// crea un flusso di uscita collegato al socket
			DataOutputStream outToClient = new DataOutputStream(connectionSocket.getOutputStream());
			
			// legge la riga dal socket
			clientSentence = inFromClient.readLine();
			capitalizedSentence = clientSentence.toUpperCase() + '\n';
			
			// scrive la riga sul socket
			outToClient.writeBytes(capitalizedSentence);
		}
	}
}
```

---
## Programmazione socket con UDP
Nella connessione UDP non c’è una connessione tra client e server (non c’è handshaking); inoltre il mittente allega esplicitamente ad ogni pacchetto l’indirizzo IP e la porta di destinazione, spetterà al server dover estrarre l’indirizzo IP e la porta del mittente dal pacchetto ricevuto

![[Pasted image 20250404193938.png|280]]
### Interazione client/server
![[Pasted image 20250404193816.png|center|500]]

### Client Java (UDP)

```java
import java.io.*;
import java.net.*;

class UDPClient {
	public static void main(String args[]) throws Exception {
		// crea un flusso di ingresso
		BufferedReader inFromUser = new BufferedReader(new InputStreamReader(System.in));
		// crea un socket client
		DatagramSocket clientSocket = new DatagramSocket();
		// traduce il nome dell'host nell'indirizzo IP usando DNS
		InetAddress IPAddress = InetAddress.getByName("hostname");
		
		byte[] sendData = new byte[1024];
		byte[] receiveData = new byte[1024];
		String sentence = inFromUser.readLine();
		
		sendData = sentence.getBytes();
		
		// crea il datagramma con i dati da trasmettere, lunghezza
		// indirizzo IP e porta
		DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, IPAddress, 9876);
		// invia il datagramma al server
		clientSocket.send(sendPacket);
		// crea il datagramma con i dati da ricevere, lunghezza
		// indirizzo IP e porta
		DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
		// legge il datagramma dal server
		// il client rimane inattivo fino a quando riceve un pacchetto
		clientSocket.receive(receivePacket);
		
		String modifiedSentence = new String(receivePacket.getData());
		System.out.println("FROM SERVER:" + modifiedSentence);
		clientSocket.close();
```

### Server Java (UDP)

```java
import java.io.*;
import java.net.*;

class UDPServer {
	public static void main(String args[]) throws Exception {
		// crea un socket per datagrammi sulla porta 9876
		DatagramSocket serverSocket = new DatagramSocket(9876);
		
		byte[] receiveData = new byte[1024];
		byte[] sendData = new byte[1024];
		
		while(true) {
			// crea lo spazio per i datagrammi
			DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
			// riceve i datagrammi
			serverSocket.receive(receivePacket);
			
			String sentence = new String(receivePacket.getData());
			
			// ottiene indirizzo IP e numero di porta del mittente
			InetAddress IPAddress = receivePacket.getAddress();
			int port = receivePacket.getPort();
			
			String capitalizedSentence = sentence.toUpperCase();
			sendData = capitalizedSentence.getBytes();
			
			// crea il datagramma da inviare al client
			DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, IPAddress, port);
			// scrive il datagramma sulla socket
			serverSocket.send(sendPacket);
		}
	}
}
```
---
Created: 2025-05-05
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Programmazione di sistema]]"
---
---
## PIPE e FIFO
Unix mette a disposizione due tipi di **Inter Process Communication** (*IPC*): **fifo** (o *named pipe*), per far comunicare processi non imparentati, e **pipe** (o *unamed pipe*), per far comunicare processi con un antenato in comune

La scrittura dei dati su una pipe/fifo (e quindi anche la lettura) avviene in maniera sequenziale (first-in first-out)

La fifo è uno speciale tipo di file che può essere creato per mezzo delle system call `mkfifo` o `mknod` (può essere utilizzata da più processi)
La pipe invece è una struttura dati in memoria *half-duplex* (la comunicazione è unidirezionale, un processo scrive l’altro legge). La creazione della pipe può essere effettuata con la system call `pipe` che crea due file descriptor (uno in lettura e uno in scrittura). Ad esempio un processo crea una pipe, poi crea un figlio e usa la pipe per comunicare con il figlio che eredita i descrittori dei file

## Pipe
Le pipe sono usato per IPC e, come già detto, sono unidireziali in particolare tra un processo padre e un figlio

![[Pasted image 20250506215825.png|center|250]]

Quando il processo padre invoca le 2 `pipe` per creare un canale di comunicazione bidirezionale e fa una `fork`, il processo figlio eredita tutti e 4 i file descriptor (2 `pipe` ognuna delle quali crea 2 `fd`, uno per la lettura e uno per la scrittura), in questo modo entrambi i processi possono leggere e scrivere

Stato delle connessioni alle `pipe1` e `pipe2` dopo la `fork`:
![[Pasted image 20250506215731.png|450]]

In questo modo però, se un processo legge e scrive dalla stessa pipe, rischia di leggere i propri dati (non si ha una vera comunicazione), per questo motivo è necessario limitare la scrittura e la lettura del processo padre e figlio:
- il padre scrive in `pipe1` e il figlio legge da `pipe1` → viene chiusa la lettura su `pipe1` e la scrittura su `pipe2` per il padre
- il figlio scrive in `pipe2` e il padre legge da `pipe2`→ viene chiusa la lettura su `pipe2` e la scrittura su `pipe1` per il figlio

Stato delle connessioni alle `pipe1` e `pipe2` dopo la chiusura appropriata dei canali di read e write:
![[Pasted image 20250506215949.png|450]]
### $\verb|pipe|$

```c
int pipe(int pipefd[2])
```

- `pipefd[0]` → file descriptor di input
- `pipefd[1]` → file descriptor di output

I dati scritti sulla pipe sono bufferizzati dal kernel finché non sono letti e le pipe hanno una dimensione massima definita dal sistema

Comportamento:
- se un processo legge da una pipe vuota allora rimane bloccato
- se un processo scrive su una pipe piena allora rimane bloccato
- una pipe viene chiusa quando tutti e due i processi hanno invocato la `close`
- operazioni di lettura (`read`) su una pipe il cui `fd` di scrittura è stato chiuso con `close` ritorna $0$
- operazioni di scrittura (`write`) su una pipe il cui `fd` di lettura è stato chiuso con `close` ritornano $-1$ e ricevono il segnale `SIGPIPE`

---
## FIFO

```c
int mkfifo(const char *pathname, mode_t mode);
```

La sysacall `mkfifo` crea una fifo con nome `pathname` e modalità di accesso `mode`
Una volta creata una fifo, può essere acceduta da qualsiasi processo (che abbia i diritti per l’accesso al file) in lettura/scrittura, tuttavia, le operazioni devono essere simultanee

Un processo che apre la fifo in lettura rimane bloccato finchè non c’è un processo che la apre in scrittura e viceversa

`mkfifo` ritorna $0$ in caso di successo e $-1$ in caso di errore impostando `errno` di conseguenza. Una volta creata può essere gestita come qualsiasi altro file

E’ inoltre possibile aprire una fifo in maniera non bloccante passando i flags `O_NONBLOCK` alla syscall `open`

---
## Socket
I socket consentono la comunicazione tra processi nel paradigma client-server. Ognuno dei due terminali ha compiti ben diversi

Server:
- definisce il socket
- il riferimento (nome di file o indirizzo di rete) è noto al client
- accetta connessioni sul socket da parte di uno o più client → ricava il full descriptor (*full-duplex*) sulla connessione

Client:
- definisce il socket
- crea una connessione sul socket → ricava un file descriptor (*full duplex*) sulla connessione

### Syscall di interesse
- `socket` → crea struttura dati della socket
- `bind` → associa un nome alla socket
- `listen` → mette un processo in ascolto su una socket
- `accept` → accetta una connessione su di una socket

### Tipologia di socket
Le socket sono definite da 3 attributi:
- **domain** (o family) → modalità del collegamento
	- `AF_LOCAL` (o `AF_UNIX`) → client e server devono risiedere sulla stessa maccina
	- `AF_INET` → client e server comunicano in rete con protocollo IPv4
	- `AF_INET6` → client e server comunicano in rete con protocollo IPv6
- **type** → semantica del collegamento
	- `SOCK_STREAM` → flusso bidirezionale di byte affidabile basato su connessione (socket TCP); supporta notifiche asincrone (*out of band*)
	- `SOCK_DGRAM` → supporta comunicazioni datagram, senza connessione (socket UDP) e con sequenze inaffidabili di messaggi
	- `SOCK_RAW` → per l’accesso diretto (raw socket) alle comunicazioni di rete (protocolli ed interfacce)
- **protocol** → protocollo usato
	- per noi ne esiste solo uno, UDP per `SOCK_DGRAM`, TCP per `SOCK_STREAM`
	- possiamo impostarlo a $0$

### Anatomia server TCP
![[Pasted image 20250507105054.png|center|300]]
#### $\verb|psrv|$
1. `psrv` definisce il socket invocando `socket()`, che crea un **unnamed socket**, ovvero una struttura dati che rappresenta il socket ma al quale non è associato un nome (indirizzo)
2. `psrv` associa un nome al socket invocando `bind()`, che trasforma l’unnamed socket in un **named socket**
3. `psrv` definisce la lunghezza della coda di ingresso invocando `listen()` (il numero massimo gestibile di connessioni pending)
4. `psrv` si mette in ascolto sulla socket in attesa di una richiesta di connessione del client invocando `accept()`, che ritorna un `fd` usato per comunicare con il client
5. `psrv` crea un figlio che userà `fd` per comunicare con il client (fa un `fork` così da poter gestire più client in contemporanea)
6. `psrv` si rimette in ascolto sulla socket con `accept()`per una nuova connessione

#### $\verb|pcli|$
1. `pcli` definisce il socket invocando `socket()`, che crea un **unnamed socket** (`csd`), ovvero una struttura dati che rappresenta il socket ma al quale non è associato un nome (indirizzo)
2. `pcli` imposta una struttura dati `sin` di tipo `struct sockaddr in` in modo da scriverci le informazioni del server al quale si vuole connettere
3. `pcli` si connette al server invocando la syscall `connect()` alla quale passa il socket (`csd`) e l’indirizzo del server al quale connettersi, e che ritorna un `fd` del server (`ssd`) oppure $-1$ in caso di errore
4. `pcli` utilizza `ssd` per leggere e scrivere da/su server
5. finita la necessità di comunicare con il server, `pcli` chiude la connessione invocando `close(ssd)`

### Struttura codice server

```c
int main() {
	int sd = socket(AF_INET, SOCKET_STREAM, 0);
	bind(sd, ...);
	listen(sd, MAX_QUEUED);
	// disabilito il segnale SIGCHLD per evitare di creare zombie process
	while (1) {
		int client sd=accpet(sd, ...);
		if (client_sd==-1) {
			perror("Errore accettando connessione dal client");
			continue;
		}
		if (fork()==0) { // eseguito dal client
			// read/write su client_sd
			close(client_sd);
			exit(0);
		}
	}
}
```

### Struttura codice client

```c
int main (){
	int cfd;
	int cfd = socket(AF_INET, SOCK_STREAM,0);
	// set sockaddr_in structure
	if (connect(cfd,….)!=0) {
		perror(“connessione non riuscita”);
	}
	// read(cfd,…) e write(cfd,…) da/verso server
	close(cfd);	
}
```

### $\verb|socket|$

```c
#include <sys/types.h>
#include ~sys/socket.h>
int socket(int domain, int type, int protocol);
```

La syscall `socket` ritorna $-1$ in caso di errore e la scrittura/lettura su un socket chiuso genera un errore **`SIGPIPE`**

> [!example]
>```c
>int sd;
>sd=socket(AF_UNIX, SOCK_STREAM, 0);
>if (sd==-1) {
>	perror("Errore creando socket");
>	exit(-1);
>}
>```

### $\verb|bind|$

```c
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```

- `sockfd` → id unnamed socket
- `addr` → struttura di tipo `sockaddr` che contiene l’indirizzo IP/nome della socket (`socaddr_in` per `AF_INET` e `AF_INET6`, `socaddr_un` per `AF_LOCAL`)
- `addrlen` → dimensione struttura `sockaddr`

>[!hint]
>Solo `AF_INET` socket possono fare binding su `IP_address:porta`

#### $\verb|sockaddr_in|$

```c
struct sockaddr_in {
	sa_family_t sin_family; /* address family: AF_INET */
	in_port_t sin_port; /* port in network byte order */
	struct in_addr sin_addr; /* internet address */
};
/* Internet address. */
struct in_addr {
	uint32_t s_addr; /* address in network byte order */
};
```

Sia il valore di `sin_port` che quello di `sin_addr` sono binari in formato **network byte order** (*NBO*)

>[!info] Network Byte Order e Host Byte Order (i386)
>- Network Byte Order → byte più significativo prima
>- Host Byte Order → byte meno significativo prima
>
>>[!example] `0x12345678`
>>- NBO → `[12] [34] [56] [78]`
>>- HBO → `[78] [56] [34] [12]`

I numeri di porta del protocollo e gli indirizzi internet vanno quindi tradotti in formato NBO. E’ inoltre possibile assegnare un valore a `sin_addr` assegnandogli una delle seguenti macro:
- `INADDR_LOOPBACK` → $127.0.0.1$ (locale)
- `INADDR_ANY` → $0.0.0.0$ (qualsiasi IPa)

### Funzioni per convertire indirizzi

```c
uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
```
`htonl` converte un `unsigned int` a $32$ bit in formato NBO
`htons` converte un `unsigned int` a $16$ bit in formato NBO

```c
int inet_aton(const char *cp, struct in_addr *inp);
char *inet_ntoa(struct in_addr in);
```
`inet_aton` converte un indirizzo `cp` di tipo ottale puntato (stringa), in formato NBO (`stuct in_addr`)
`inet_ntoa` converte un indirizzo `in` di tipo NBO, in formato ottale puntato (stringa)

```c
struct hostent *gethostbyname(const char *name);
```
Dato un nome logico (`mio.dominio.toplevel`) o un indirizzo in formato ottale puntato ritorna una struttura `hostent` che contiene l’indirizzo in formato NBO (se riceve un IP fa la stessa cosa di `inet_aton`, mentre se riceve un hostname fa una query DNS)

### $\verb|listen|$

```c
int listen(int sockfd, int backlog);
```

Marca il socket `sockfd` come passive, ovvero pronto a ricevere richieste mediante una `accept()`. `backlog` invece indica la lunghezza della coda di attesa

Restituisce $0$ in caso di successo e $-1$ in caso di errore

### $\verb|accept|$

```c
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
```

Usata per i socket con connessione (non per `SOCK_DGRAM`) e serve ad estrarre la prima richiesta di connessione nella coda delle richieste in attesa sulla coda si listening del socket `sockfd`

Questa inoltre crea un nuovo socket con connessione e ritorna un nuovo `fd` (il nuovo socket non è in ascolto)
---
Created: 2024-11-08
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Categorie di dispositivi I/O
L’I/O è un’area assai problematica nella progettazione di sistemi operativi in quanto ci sta una grande varietà di dispositivi I/O e anche una grande varietà di applicazioni che li usano, risulta quindi difficile scrivere un SO che tanga conto di tutto

Esistono tre categorie di dispositivi:
- leggibili dall’utente
- leggibili dalla macchina
- dispositivi di comunicazione

### Dispositivi leggibili dall’utente
Sono tutti quei dispositivi usati per la comunicazione diretta con l’utente, come ad esempio:
- stampanti
- “terminali” → monitor, tastiera, mouse, ecc. (da non confondere con i terminali software)
- joystick
- ecc.

### Dispositivi leggibili dalla macchina
Sono quei dispositivi usati per la comunicazione con materiale elettronico, come ad esempio:
- dischi
- chiavi USB
- snesopri
- controllori
- attuatori

### Dispositivi per la comunicazione
Dispositivi usati per la comunicazione con dispositivi remoti, come ad esempio:
- modem
- schede Ethernet
- Wi-Fi

---
## Funzionamento (semplificato) dei dispositivi I/O
Un dispositivo di *input* prevede di essere **interrogato sula valore di una certa grandezza fisica al suo interno** (eg. per il mouse si tratta delle coordinate dell’ultimo spostamento effettuato, per la tastiera il codice Unicode dei tasti premuti, per il disco il valore dei bit che si trovano in una certa posizione al suo interno).
L’idea è che se un processo effettua una syscall `read` su un dispositivo del genere, vuole conoscere questo dato (per poterlo elaborare e fare altro)

Un dispositivo di *output* prevede di poter **cambiare il valore di una certa grandezza fisica al suo interno** (eg. per un monitor il valore RGB dei pixel, per una stampante il PDF di un file da stampare, per il disco il valore dei bit che devono sovrascrivere quelli che si trovano in una certa posizione al suo interno)
L’idea è che se un processo effettua una syscall `write` su un dispositivo del genere, vuole cambiare qualcosa (spesso l’effetto è direttamente visibile all’utente, ma in altri casi è visibile solo usando altre funzionalità di lettura, come per il disco)ù

Ci sono quindi almeno due syscall **`read`** e **`write`** (tra i dati che prendono in input queste syscall ci sta un identificativo del dispositivo su  cui si vuole eseguire l’operazione, in Linux sono i *file descriptor*)
Dunque quando arriva una di queste due syscall, il kernel comanda l’inizializzazione del trasferimento di informazioni, mette il processo in blocked e passa ad altro
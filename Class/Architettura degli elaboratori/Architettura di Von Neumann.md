---
Created: 2024-03-05
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

>[!info] Index
>- [[#Struttura|Struttura]]
>- [[#CPU|CPU]]
>- [[#Memoria|Memoria]]
>- [[#Dispositivi I/O|Dispositivi I/O]]

---
## Struttura

Il modello di architettura di von Neumann è costituito da tre sottosistemi interconnessi: 
- **[[#CPU]]** (Central Processing Unit - Processore)
- **[[#Memoria]]**
- **[[#Dispositivi I/O]]** (Input/Output o Ingresso/Uscita).  

La connessione tra questi sottosistemi è realizzata mediante i **Bus**.

![[Pasted image 20240305151702.png|600]]



**Compiti:**
- Elaborazione (CPU)
- Memorizzazione (Memorie)
- Scambio di informazioni (che avviene attraverso I/O)
- Controllo (che sincronizza tutti i compiti) (svolto da CPU )

---
## CPU
Il processore (CPU, Central Processing Unit) è un interprete di istruzioni, costituito da tre componenti:

- **ALU** (Unità Aritmetico Logica): esegue le operazioni matematiche e logiche.
- **Unità di Controllo**: legge le istruzioni, le decodifica e le esegue ed effettua i controlli delle attività necessarie per l’esecuzione
- **Registri:** sono molto veloci e con una capacità ridotta, costituiscono una memoria speciale (di supporto) per l’ALU poiché contengono le istruzioni di controllo necessarie per il suo funzionamento e i risultati temporanei delle elaborazioni.
- **Bus:**
	- bus di comunicazione con la memoria 
	- bus di comunicazione con periferiche (non necessario con memory mapping)

![[Pasted image 20240305121228.png|300]]

---
## Memoria

***NON SPIEGATO DAL PROF***

La Memoria contiene i dati e i programmi e la sua capacità è espressa in multipli del Byte.  
Il Byte è una sequenza di otto bit, che insieme rappresentano un singolo carattere alfabetico e/o numerico. Le dimensioni della memoria sono espresse come multipli molto più grandi:  
– Kilobytes (1.024 bytes),  
– Megabytes (1.024 Kilobytes),  
– GigaBytes (1.024 Megabytes),  
– TeraBytes (1.024 Gigabytes).

I dispositivi di memoria possono essere suddivisi in più classi, in dipendenza della capacità e della velocità. Esistono due classi principali: la memoria centrale e la memoria secondaria.

La **memoria Centrale** ha una funzione di supporto alla CPU perché fornisce (ad alta velocità) le istruzioni del programma in esecuzione e i dati su cui operare. È composta da un insieme di locazioni (celle), ciascuna delle quali può memorizzare una parte delle informazioni. Ad ogni locazione è associato un indirizzo (ossia un numero che la identifica univocamente). La memoria centrale si suddivide in due componenti:

- **ROM** (Read Only Memory): memoria di sola lettura, cioè i dati non sono modificabili dall’utente. È una memoria permanente (conserva le informazioni anche dopo lo spegnimento del computer) e contiene i programmi fondamentali per l’avvio del computer, noti come BIOS (che interagiscono con i circuiti della macchina).
- **RAM** (Random Access Memory): memoria ad accesso casuale e di tipo volatile, cioè il suo contenuto va perso quando si spegne il computer. Contiene i dati (intermedi e finali delle elaborazioni) e le istruzioni dei programmi in esecuzione.

La **memoria EPROM** (Electric Programmable ROM) è paragonabile alla memoria ROM cui si è accennato in precedenza, ma, diversamente da quest’ultima, consente in particolari condizioni la modifica dei dati in essa contenuti. Ovviamente, qualsiasi modifica operata determina sostanziali modifiche nel funzionamento del computer, per cui la stessa non può essere oggetto di improvvisazione e deve essere affidata soltanto ad utenti esperti.

La **memoria CACHE** è invece destinata ad ospitare dati di frequente utilizzo, e consente un accesso più veloce a informazioni di recente acquisite e visualizzate; è il caso, ad esempio, dei dati cui si ha avuto accesso per mezzo di Internet. È una memoria molto utile e può essere “svuotata” a piacimento dall’utente, al fine di renderla disponibile per ulteriori archiviazioni temporanee.

La **Memoria secondaria** (o di massa) è più lenta, ha una elevata capacità di immagazzinare i dati (di uso non frequente) ed è stabile, ossia mantiene la memorizzazione delle informazioni anche dopo lo spegnimento del computer, per questo è utilizzata per la memorizzazione permanente di dati e programmi.

---
## Dispositivi I/O

I Dispositivi di Input/Output (o periferiche), sotto il controllo e coordinamento del processore, consentono l’interazione tra il computer e l’utente (più in generale, l’interazione tra il computer e l’ambiente), in particolare consentono l’immissione dei dati all’interno del computer e la comunicazione all’esterno dei risultati ottenuti con l’elaborazione.

---
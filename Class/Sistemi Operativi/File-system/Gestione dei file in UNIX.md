---
Created: 2024-11-24
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
In Unix ci sono sei tipi di file:
- normale
- directory
- speciale (mappano su nomi di file i dispositivi di I/O)
- named pipe (per far comunicare i processi tra loro)
- hard link (collegamenti, nome di file alternativo)
- link simbolico (il suo contenuto è il nome del file cui si riferisce)

---
## Inode
Inode sta per *”index node”* (ispirato al metodo di [[Gestione della memoria secondaria#Allocazione indicizzata#Porzioni di lunghezza fissa|allocazione indicizzato]], con dimensione fissa dei blocchi) e rappresentano i [[Gestione della memoria secondaria#Dati e metadati|metadati]] di un file
Questa struttura dati contiene le informazioni essenziali per un dato file (che rientra in quelli sopra elencati, anche la directory) e un numero che permette di accedere più facilmente ad un inode (inode number, quando si cancella un file l’inode number può essere riutilizzato).

>[!info]
>Un dato inode potrebbe essere associato a più nomi di file (anche se di solito un inode, un file) tramite un hard link

Tutti gli inode si trovano in una zona di disco dedicata (*i-list*), viene però mantenuta dal SO una tabella di tutti gli inode corrispondenti a file aperti in memoria principale
### Inode in Free BSD
 All’interno di un inode sono contenuti:
 - Tipo e modo di accesso del file
- Identificatore dell’utente proprietario e del gruppo cui tale utente appartiene
- Tempo di creazione e di ultimo accesso (lettura o scrittura)
- Flag utente e flag per il kernel
- Numero sequenziale di generazione del file
- Dimensione delle informazioni aggiuntive
- Altri attributi (controllo di accesso e altro)
- Dimensione
- Numero di blocchi, o numero di file (per le directory)
- Dimensione dei blocchi
- Sequenze di puntatori a blocchi

![[Pasted image 20241124191720.png]]

Per file piccoli (di grandezza massima di $13\cdot \text{dimensione di un blocco}$), i dati sono puntati direttamente da `direct`
Infatti i blocchi `direct` puntano ai blocchi stessi contenenti i dati, mentre i puntatori indiretti puntano ai cosiddetti **”blocchi di indirizzamento”** ovvero quei blocchi in cui sono contenuti gli indirizzi dei blocchi in cui si trovano i dati veri e propri. Questi vengono utilizzati sono nel momento in cui i puntatori diretti non sono sufficienti

---
## Allocazione di file
L’allocazione di file è dinamica e viene fatta a blocchi (quindi potenzialmente non contigui). Attraverso l’indicizzazione si tiene traccia dei blocchi dei file (parte dell’indice è puntata dall’inode) all’interno dei puntatori indiretti, mentre si punta ai dati veri e propri attraverso i `direct` (per file piccoli)

![[Pasted image 20241124194202.png]]

---
## Inode e directory
Anche le directory sono file. Infatti al loro interno è contenuta una lista di coppie (nome file, puntatore ad inode), ma alcuni di questi file possono essere a loro volta delle directory, formando così una struttura gerarchica

Una directory può essere letta da ogni utente ma modificata solo dal sistema operativo

![[Pasted image 20241124194521.png]]

---
## Accesso ai file
Per ogni file, ci sono 3 terne di permessi: lettura, scrittura ed esecuzione. Di queste terne ce ne sta una per il proprietario, una per il suo gruppo (insieme di utenti) e una per tutti gli altri

![[Pasted image 20241124195008.png]]

---

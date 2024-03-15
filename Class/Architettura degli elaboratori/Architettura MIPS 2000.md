---
Created: 2024-03-06
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

>[!info] Index
>- [[#Introduzione|Introduzione]]
>- [[#MIPS 2000|MIPS 2000]]

---
## Introduzione
In era moderna, possiamo individuare due tipologie principali di architetture di
calcolatori:

| CISC                                                                                                                                                              | RISC                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Istruzioni di dimensioni variabile**<br>- Per il fetch della successiva è necessaria la decodifica della precedente                                             | **Istruzioni di dimensione fissa**<br>- Fetch della successiva senza decodifica della precedente                                                |
| **Formato variabile**<br>- Decodifica complessa                                                                                                                   | **Istruzioni di formato uniforme**<br>- Per semplificare la fase di decodifica                                                                  |
| **Operandi in memoria**<br>- Molti accessi alla memoria per istruzione                                                                                            | **Operazioni ALU solo tra registri**<br>- Senza accesso a memoria                                                                               |
| **Pochi registri interni**<br>- Maggior numero di accessi in memoria                                                                                              | **Molti registri interni**<br>- Per i risultati parziali senza accessi alla memoria                                                             |
| **Modi di indirizzamento complessi**<br>- Maggior numero di accessi in memoria<br>- Durata variabile dell’istruzione<br>- Conflitti tra istruzioni più complicati | **Modi di indirizzamento semplici**<br>- Con spiazzamento (un solo accesso a memoria)<br>- Durata fissa dell’istruzione<br>- Conflitti semplici |
| **Istruzioni complesse**<br>- Pipeline più complicata<br> - Più veloci a svolgere operazioni complesse                                                            | **Istruzioni semplici**<br>- Pipeline più veloce<br>- Più lento nello svolgere operazioni complesse                                             |
Riassumendo possiamo dire che le **Architetture CISC** risultano più complesse ma ottimizzate per singoli scopi, mentre le **Architetture RISC**, in quanto più semplici, risultano adatte a scopi generici

## MIPS 2000
![[Screenshot 2024-03-14 alle 21.48.33.png|center|400]]
Per via delle sue caratteristiche l’**Architettura MIPS** (Microprocessor without Interlocked Pipelined Stages), risiede all’interno delle Architetture RISC
In particolare, l’**Architettura MIPS 2000** è composta da:
- Tutte le **word hanno una dimensione fissa** di 32 bit
- Lo **spazio di indirizzamento** è di $2^{30}$ word di 32 bit ciascuna, per un totale di 4 GB
- Una **memoria indicizzata al byte**, dunque, dato un indirizzo di memoria $t$ corrispondente all’inizio di una word, per leggere la word successiva è necessario utilizzare l’indirizzo $t+4$, poiché 4 byte corrispondono a 32 bit
- Gli interi vengono salvati utilizzando la notazione del **Complemento a 2** su 32 bit
- Dotata di **3 microprocessori**
	- La **CPU principale**, dotata di ALU, di 32 registri HI/LO ed addetta all’esecuzione delle istruzioni
	- Il **Coprocessore 0**, non è dotato di registri e non ha accesso alla memoria, ma è solo addetto alla gestione di "trap", eccezioni, Virtual Memory, Cause, EPC, Status, BadVAddr, …
	- Il **Coprocessore 1**, addetto ai calcoli in virgola mobile e dotato di 32 registri da 32 bit, utilizzabili anche come 64 registri da 16 bit
- I **32 Registri**, indicizzati da 0 a 31, della CPU principale
	- Registro **$zero** ($0) → contenente un valore costante pari a 0 ed immutabile
	- Registro **$at** ($1) → usato dalle pseudoistruzioni e dall’assemblatore
	- Registri **$v0 e $v1** ($2, $3) → utilizzati per gli output delle funzioni utilizzate nel programma
	- Registri **dall’\$a0 all’$a3** ($4, ..., $7) → utilizzati per gli input delle funzioni
	- Registri **dal $t0 al $t7** ($8, ..., $15) → utilizzati per valori temporanei
	- Registri **dal $s0 al $s7** ($16, ..., $23) → utilizzati per valori più ricorrenti
	- Registri **dal $t8 al $t9** ($24, $25) → utilizzati per valori temporanei
	- Registri **$k0 e $k1** ($26, $27) → utilizzati dal Kernel del Sistema Operativo
	- Registro **$gp** ($28) → ossia Global Pointer, utilizzato per la gestione della memoria dinamica
	- Registro **$sp** ($29) → ossia Stack Pointer, utilizzato per la gestione dello Stack delle funzioni
	- Registro **$fp** ($30) → ossia Frame Pointer, utilizzato dalle funzioni di sistema
	- Registro **$ra** ($31) → ossia Return Address, utilizzato come puntatore di ritorno dalle funzioni
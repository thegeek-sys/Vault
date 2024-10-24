---
Created: 2024-04-15
Class: "[[Architettura degli elaboratori]]"
Related:
  - "[[Rappresentazione dell’informazione]]"
Completed:
---
---

>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#Progettare la CPU MIPS|Progettare la CPU MIPS]]
>- [[#Unità funzionali necessarie|Unità funzionali necessarie]]
>- [[#Ingredienti|Ingredienti]]
>	- [[#Ingredienti#Memoria delle istruzioni, PC, adder|Memoria delle istruzioni, PC, adder]]
>	- [[#Ingredienti#Registri e ALU|Registri e ALU]]
>	- [[#Ingredienti#Memoria dati ed unità di estensione del segno|Memoria dati ed unità di estensione del segno]]
>- [[#Fetch dell’istruzione/aggiornamento PC|Fetch dell’istruzione/aggiornamento PC]]
>- [[#Operazioni ALU e accesso a MEM|Operazioni ALU e accesso a MEM]]
>- [[#Esercizi|Esercizi]]
>- [[#Salti condizionati (beq)|Salti condizionati (beq)]]
>- [[#Tutto insieme|Tutto insieme]]
>- [[#…con Control Unit e logica di salto|…con Control Unit e logica di salto]]
>- [[#Formato delle istruzioni e bit di controllo ALU|Formato delle istruzioni e bit di controllo ALU]]
>	- [[#Formato delle istruzioni e bit di controllo ALU#Input di controllo e tabella di verità|Input di controllo e tabella di verità]]
>- [[#Datapath completo|Datapath completo]]
>- [[#Segnali di controllo|Segnali di controllo]]
>- [[#Segnali da generare|Segnali da generare]]
>- [[#Tempi di esecuzione|Tempi di esecuzione]]
>	- [[#Tempi di esecuzione#Esempio|Esempio]]

---
## Introduction
$$
\text{CPU} = \text{macchina sequeniale} \to \text{Stato}+\text{Circuito combinatorio}
$$
Per **stato** si intende un insieme di parametri che se inseriti nella macchina sequenziale i restituisce uno stesso output.

---
## Progettare la CPU MIPS
Prima fase: CPU MIPS semplice non ottimizzata (senza pipeline)
- definire come viene elaborata una istruzione (**fasi di esecuzione**)
- scegliere le **istruzioni da realizzare**
- scegliere le **unità fondamentali necessarie**
- **collegare** le unità funzionali
- **costruire la Control Unit** che controlla il funzionamento della CPU
	fa in modo che l’istruzione letta venga eseguita in modo corretto, sa quali parti della CPU attivare data una determinata sequenza binaria
- calcolare il massimo tempo di esecuzione delle istruzioni (che ci dà il **periodo del clock**)

Fasi di esecuzione di una istruzione:
- *fetch* → **caricamento** di una istruzione dalla memoria alla CU
- *decodifica* → **decodifica** della istruzione e **lettura argomenti** dai registri
- *esecuzione* → **esecuzione** (attivazione delle unità funzionali necessarie)
- *memoria* → accesso alla **memoria**
- *write back* → scrittura dei **risultati nei registri**
Altre operazioni necessarie:
- aggiornamento del PC (normale, salti condizionati, salti non condizionati)

> [!hint]
> La cosa particolare sta nel fatto che i passaggi da eseguire per una singola istruzione li posso fare in parallelo. Se ad esempio faccio un `add $s1,$t0,$t1` posso contemporaneamente leggere i valori di $t1 e $t0 e poi in un secondo momento fare la somma

Istruzioni da realizzare:
- **accesso alla memoria** → `lw, sw` (tipo I)
- **salti condizionati** → `beq` (tipo I)
- **operazioni aritmetico-logiche** → `add, sun, sll, slt` (tipo R)
- **salti non condizionati** → `j, jal` (tipo J)
- **operandi non costanti** → `li, addi, subi` (tipo I)

Formato delle istruzioni MIPS (ovvero la loro codifica in memoria)
![[Screenshot 2024-04-15 alle 16.09.41.png]]

---
## Unità funzionali necessarie
- **PC** → registro che contiene l’indirizzo della istruzione
- **memoria istruzioni** → contiene le istruzioni
- **memoria dati** → da cui leggere/in cui scrivere i dati (load/store)
- **adder** → per calcolare il PC (successivo o salto)
- **registri** → contengono gli argomenti delle istruzioni
- **ALU** → fa le operazioni aritmetico-logiche, confronti, indirizzi in memoria

Le varie unità funzionali sono collegate tra loro attraverso diversi **datapath** (interconnessioni che definiscono il flusso delle informazioni nella CPU, i cavi)

Se un’unità funzionale può ricevere dati da **più sorgenti** è necessario inserire un multiplexer (**MUX**) per selezionare la sorgente necessaria

Le unità funzionali sono attivate e coordinate dai segnali prodotti dalla **Control Unit**

---
## Ingredienti
### Memoria delle istruzioni, PC, adder
**Memoria istruzioni**:
- Input → indirizzo a 32 bit
- Output → istruzione (da 32 bit) situata nell’indirizzo di input

**Program counter**:
- Registro che contiene l’**indirizzo** dell’istruzione corrente

**Sommatore**:
- Necessario per calcolare il nuovo PC e le destinazioni dei salti relativi
- riceve due valori a 32 bit e ne fornisce in uscita la somma

![[Screenshot 2024-04-15 alle 16.44.10.png]]


### Registri e ALU
**Blocco dei registri** (register file):
- contiene **32 registri** a 32 bit, indirizzabili con 5 bit ($2^5 = 32$)
- può memorizzare un dato in un registro e contemporaneamente fornirlo in uscita
- 3 porte a 5 bit per indicare quali 2 registro scrivere
- 3 porte dati (a 32 bit)
	- una in ingresso per il valore da memorizzare
	- 2 di uscita per i valori letti
- il segnale *RegWrite* abilita (se 1) la scrittura nel registro di scrittura

**ALU**:
- riceve due valori interi a 32  bit e svolge una operazione indicata dai segnali *Op. ALU*
- oltre al risultato da 32 bit e produce un segnale `Zero` asserito se il risultato è zero

![[Screenshot 2024-04-15 alle 16.52.18.png]]


### Memoria dati ed unità di estensione del segno
**Unità di memoria**:
- riceve un **indirizzo** (da 32 bit) che indica quale word della memoria va letta/scritta
- riceve il segnale **MemRead** che abilita la lettura dall’indirizzo (`lw`)
- riceve un dato da 32 bit da scrivere in memoria a quell’indirizzo (`sw`)
- riceve il segnale di controllo **MemWrite** che abilita (1) la scrittura del dato all’indirizzo
- fornisce su una porta di uscita da 32 bit il lato letto (se MemRead = 1)

**L’unità di estensione del segno**:
- serve a trasformare un intero relativo (in CA2) da 16 a 32 bit, ovvero copia il bit del segno nei 16 bit più significativi della parola

![[Screenshot 2024-04-15 alle 17.02.07.png]]

---
## Fetch dell’istruzione/aggiornamento PC
1. PC = indirizzo dell’istruzione
2. Lettura dell’istruzione
3. PC incrementato di 4 (1 word)
4. Valore aggiornato e reimmesso nel PC

![[Screenshot 2024-04-15 alle 17.04.20.png|center|400]]

> [!info]
> - Connessioni da 32 bit
> - Mentre viene letta l’istruzione viene già calcolato il nuovo PC

---
## Operazioni ALU e accesso a MEM
Decodifica facilitata: i formati I ed R sono quasi uguali
Secondo argomento dell’istruzione (a seconda del segnale di controllo **ALUSrc** che seleziona la porta corrispondente del MUX):
- registro
- campo immediato (in questo caso: valore esteso nel segno)

Per calcolare l’indirizzo di accesso alla memoria si usa la stessa ALU (reg. base + campo i.). Il risultato dell’ALU o della `lw` viene selezionato da **MemtoReg** che comanda il MUX a destra

---
## Esercizi
![[Screenshot 2024-04-15 alle 17.17.55.jpg]]

![[Screenshot 2024-04-15 alle 17.38.13.jpg]]

![[Screenshot 2024-04-15 alle 17.53.42.jpg]]

---
## Salti condizionati (beq)
**ALU** come **comparatore** (sottrazione) di cui il segnale `Zero` ci indica se operare il salto
La destinazione dei salti è un **numero relativo di istruzioni** rispetto alla istruzione seguente estesa del segno, moltiplicata per 4, sommata a PC + 4

Il nuovo valore del PC può dunque provenire da:
- PC+4 → istruzione seguente
- uscita dell’adder → salto

>[!info] In questo caso non ho bisogno di alcun MUX

![[Screenshot 2024-04-16 alle 18.59.46.png]]

---
## Tutto insieme
![[Screenshot 2024-04-16 alle 19.33.04.png]]

>[!info]- Esercizio
>Evidenziare le linee attraversate dai segnali di `lw $s0,0x00000004($29)`
>![[Screenshot 2024-04-16 alle 19.33.04.jpg]]

## …con Control Unit e logica di salto
![[Screenshot 2024-04-16 alle 19.47.13.png]]

---
## Formato delle istruzioni e bit di controllo ALU
Fino ad ora abbiamo considerato i **4 bit** di controllo dell’operazione da eseguire dall’ALU come delle incognite. L’ALU in realtà fa un totale di 6 operazioni in base alla seguente codifica

| ALU control lines |     Function     |
| :---------------: | :--------------: |
|       0000        |       AND        |
|       0001        |        OR        |
|       0010        |       add        |
|       0110        |     subract      |
|       0111        | set on less than |
|       1100        |       NOR        |

Questi 4 bit più altri due, dati dall’**ALUOp**, formano l’**OpCode** dell’istruzione.
Ciò permette di notare immediatamente che tipo di operazione, infatti se l’ALUOp inizia per 0 si tratta di operazioni `lw`, `sw` o `beq` (in cui non necessito dei 6 bit di funct) altrimenti di operazioni di tipo R.

Per questo motivo si hanno due livelli di decodifica
- un primo eseguito dalla **Control Unit**
- un secondo eseguito dall’**ALU** (che riceve in input i 4 bit di controllo dell’operazione di cui abbiamo parlato sopra)

| Codice operativo istruzione | ALUOp | Operazione eseguita dall’istruzione | Campo funzione | Operazione dell’ALU | Ingresso di controllo alla ALU |
|:---------------------------:|:-----:|:-----------------------------------:|:--------------:|:-------------------:|:------------------------------:|
|            `lw`             |  00   |          load di 1 parola           |     XXXXXX     |        somma        |              0010              |
|            `sw`             |  00   |          store di 1 parola          |     XXXXXX     |        somma        |              0010              |
|        Branch equal         |  01   | salto condizionato all’uguaglianza  |     XXXXXX     |     sottrazione     |              0110              |
|           Tipo R            |  10   |                somma                |     100000     |        somma        |              0010              |
|           Tipo R            |  10   |             sottrazione             |     100010     |     sottrazione     |              0110              |
|           Tipo R            |  10   |                 AND                 |     100100     |         AND         |              0000              |
|           Tipo R            |  10   |                 OR                  |     100101     |         OR          |              0001              |
|           Tipo R            |  10   |            set less than            |     101010     |    set less than    |              0111              |
### Input di controllo e tabella di verità
![[Screenshot 2024-04-16 alle 20.04.21.png]]
N.B. X rappresentano don’t care

## Datapath completo
![[Screenshot 2024-04-18 alle 22.36.47.png|700]]

>[!info]- Esecuzioni
>**Tipo R**
>![[Screenshot 2024-04-18 alle 22.39.10.png]]
>
>**lw**
>![[Screenshot 2024-04-18 alle 22.40.03.png]]
>
>**beq**
>![[Screenshot 2024-04-18 alle 22.41.23.png]]

---
## Segnali di controllo

| Nome del segnale | Effetto quando non asserito                                                                  | Effetto quando asserito                                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `RegDst`         | il numero dei registri in scrittura proviene dal campo rt (bit 20-16)                        | il numero del registro di scrittura proviene dal campo rd (bit 15-11)                                                              |
| `RegWrite`       | nulla                                                                                        | il dato viene scritto nel register file nel registro individuato dal numero del registro di scrittura                              |
| `ALUSrc`         | il secondo operando della ALU proviene dalla seconda uscita del register file (Dato letto 2) | il secondo operando della ALU proviene dall’estensione del segno dei 16 bit meno significativi dell’istruzione                     |
| `MemRead`        | nulla                                                                                        | il dato della memoria nella posizione puntata dall’indirizzo viene inviato in uscita sulla linea “dato letto”                      |
| `MemWrite`       | nulla                                                                                        | il contenuto della memoria nella posizione puntata dall’indirizzo viene sostituito con il dato presente sulla linea “dato scritto” |
| `MemtoReg`       | il dato viene inviato al register file per la scrittura, proviene dalla ALU                  | il dato inviato al register file per la scrittura proviene dalla Memoria Dati                                                      |

---
## Segnali da generare
L’ALU deve seguire 4 tipi di comportamento:
- Se l’istruzione è di **tipo R** eseguire l’operazione indicata dal campo **funct** dell’istruzione
- Se l’istruzione accede alla memoria (**lw**, **sw**) svolgere la **somma** che calcola l’indirizzo
- Se l’istruzione è un **beq** deve svolgere una **differenza**
Per codificare 3 comportamenti bastano 2 segnali dalla Control Unit: `ALUOp1` ed `ALUOp0`

| istruzione | regDst | ALUSrc | MemtoReg | RegWrite | MemRead | MemWrite | Branch | ALUOp1 | ALUOp0 |
| ---------- | :----: | :----: | :------: | :------: | :-----: | :------: | :----: | :----: | :----: |
| Tipo R     |   1    |   0    |    0     |    1     |    0    |    0     |   0    |   1    |   0    |
| `lw`       |   0    |   1    |    1     |    1     |    1    |    0     |   0    |   0    |   0    |
| `sw`       |   X    |   1    |    X     |    0     |    0    |    1     |   0    |   0    |   0    |
| `beq`      |   X    |   0    |    X     |    0     |    0    |    0     |   1    |   0    |   1    |

---
## Tempi di esecuzione
Se conosciamo il tempo necessario a produrre i risultati delle diverse unità funzionali allora possiamo calcolare il tempo totale di ciascuna istruzione.
Bisogna però fare attenzione a se le istruzioni sono eseguite in **serie** o in **parallelo**, se sono in parallelo infatti devo prendere il **massimo tra i tempi dei vari branch** che eseguono le operazioni in serie
### Esempio
Supponiamo che i tempi siano:
- accesso alla memoria (dati o istruzione) → 100 ns
- ALU e sommatori → 150 ns
- accesso ai registri (in lettura o scrittura) → 50 ns
- tutte le altre componenti → 0 ns

Allora i tempi di esecuzione delle istruzioni saranno:

| Istruzione | Istruction Fetch | Instruction Decode | Execution | MEM | Write Back | Totale |
| ---------- | ---------------- | ------------------ | --------- | --- | ---------- | ------ |
| di tipo R  | 100              | 50                 | 150       |     | 50         | 350    |
| lw         | 100              | 50                 | 150       | 100 | 50         | 450    |
| sw         | 100              | 50                 | 150       | 100 |            | 400    |
| beq        | 100              | 50                 | 150       |     |            | 500    |
|            |                  |                    |           |     |            |        |
> [!info]
> le due operazioni di somma per calcolare PC +4 (150ns) e salti condizionati (altri 150ns) sono svolte in parallelo al Fetch, Decode ed Execution e non allungano i tempi


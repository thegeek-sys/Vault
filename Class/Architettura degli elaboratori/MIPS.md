---
Created: 2024-03-06
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

>[!info] Index
>- [[#Introduzione|Introduzione]]
>- [[#Set di istruzioni|Set di istruzioni]]
>- [[#Operandi MIPS|Operandi MIPS]]
>- [[#Istruzioni linguaggio assembler MIPS|Istruzioni linguaggio assembler MIPS]]
>	- [[#Istruzioni linguaggio assembler MIPS#Aritmetiche|Aritmetiche]]
>	- [[#Istruzioni linguaggio assembler MIPS#Trasferimento di dati|Trasferimento di dati]]
>	- [[#Istruzioni linguaggio assembler MIPS#Logiche|Logiche]]
>	- [[#Istruzioni linguaggio assembler MIPS#Salti condizionali|Salti condizionali]]
>	- [[#Istruzioni linguaggio assembler MIPS#Salti incondizionati|Salti incondizionati]]

---
## Introduzione
L’architettura MIPS (**M**icroprocessor without **I**nterlocked **P**ipelined **S**tages) è un'architettura per microprocessori RISC. Il disegno dell'architettura e del set di istruzioni è semplice e lineare, spesso utilizzato come caso di studio nei corsi universitari indirizzati allo studio delle architetture dei processori.
La memoria MIPS è indicizzata al byte. Dunque se mi trovo all’indirizzo t e devo leggere la parola successiva incremento indirizzo come t+4, questo perché una parola sono 4 byte ossia 32 bit (1 byte = 8 bit)

---
## Set di istruzioni
Le fasi di esecuzione di un’istruzione sono:
- **fetch**/caricamento dell’istruzione
	dalla posizione indicata dal Program Counter (dalla RAM alla CPU) (particolarmente lento)
- **decodifica**/riconoscimento dell’istruzione
	la control unit legge i 6 bit dell’opcode e inizia a settare la CPU
- **load**/caricamento di eventuali argomenti
	leggo i registri (molto veloce)
- **esecuzione** della istruzione
	eseguita in genere dall’ALU
- **store**/salvataggio del risultato
	scrivere il risultato sulla CPU o RAM
- aggiornamento del **Program Counter**
	vado avanti nel programma o faccio un salto

Tipologie di istruzioni:
- **load/store** → trasferisco dati da/verso la memoria
- **logico/aritmetiche** → svolgono i calcoli aritmetici e logici
- **salti** condizionati e incondizionati → controllano il flusso logico di esecuzione (usati per cicli e per uscire da una sottoroutine, quando si entra in una funzione sto facendo un jump nella ram e quando esco faccio un altro jump indietro)
- gestione delle **eccezioni/interrupt** → salvataggio dello stato e suo ripristino
- istruzioni di **trasferimento dati** → non necessarie col memory-mapping

---
## Operandi MIPS

| Nome                                 | Esempio                                                                               | Commenti                                                                                                                                                                                                                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 32 registri <br>(CPU)                | `$s0-$s7`,`$t0-$t9`, `$zero`, `$a0-$a3`, `$v0-$v1`, `$gp`, `$fp`, `$sp`, `$ra`, `$at` | Accesso veloce ai dati. Nel MIPS gli operandi devono essere contenuti nei registri per potere eseguire delle operazioni. Il registro `$zero` contiene sempre il valore 0, e il registro `$at` viene riservato all’assemblatore per la gestione di costanti molto lunghe                              |
| $2^{30}$ parole in memoria <br>(RAM) | `Memoria[0]`, `Memoria[4]`, ….                                                        | Alla memoria si accede solamente attraverso le istruzioni di trasferimento dati. Il MIPS utilizza l’**indirizzamento al byte**, perciò due parole consecutive hanno indirizzi in memoria a una distanza di 4. La memoria consente di memorizzare strutture dati, vettori o il contenuto dei registri |

---
## Istruzioni linguaggio assembler MIPS
### Aritmetiche

| Istruzioni      | Esempio           | Significato       | Commenti                              |
| --------------- | ----------------- | ----------------- | ------------------------------------- |
| Somma           | `add $s1,$s2,$s3` | `$s1 = $s2 + $s3` | Operandi in tre registri              |
| Sottrazione     | `sub $s1,$s2,$s3` | `$s1 = $s2 - $s3` | Operandi in tre registri              |
| Somma immediata | `addi $s1,$s2,20` | `$s1 = $s2 + 20`  | Utilizzata per sommare delle costanti |

### Trasferimento di dati

| Istruzioni                                        | Esempio           | Significato                               | Commenti                                                                |
| ------------------------------------------------- | ----------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| Lettura parola                                    | `lw $s1,20($s2)`  | `$s1 = Memoria[$s2+20]`                   | Trasferimento di una parola da memoria a registro                       |
| Memorizzazione parola                             | `sw $s1,20($s2)`  | `Memoria[$s2+20]=$s1`                     | Trasferimento di una parola da registro a memoria                       |
| Lettura mezza parola                              | `lh $s1,20($s2)`  | `$s1=Memoria[$s2+20]`                     | Trasferimento di una mezza parola da memoria a registro                 |
| Lettura mezza parola senza segno                  | `lhu $s1,20($s2)` | `$s1=Memoria[$s2+20]`                     | Trasferimento di una mezza parola da memoria a registro                 |
| Memorizzazione mezza paola                        | `sh $s1,20($s2)`  | `Memoria[$s2+20]=$s1`                     | Trasferimento di una mezza parola da memoria a registro                 |
| Lettura byte                                      | `lb $s1,20($s2)`  | `$s1=Memoria[$s2+20]`                     | Trasferimento di un byte da memoria a registro                          |
| Lettura byte senza segno                          | `lbu $s1,20($s2)` | `$s1=Memoria[$s2+20]`                     | Trasferimento di un byte da memoria a registro                          |
| Memorizzazione byte                               | `sb $s1,20($s2)`  | `Memoria[$s2+20]=$s1`                     | Trasferimento di un byte da registro a memoria                          |
| Lettura di una parola e blocco                    | `ll $s1,20($s2)`  | `$s1=Memoria[$s2+20]`                     | Caricamento di una parola come prima fase di un’operazione atomica      |
| Memorizzazione condizionata di una parola         | `sc $s1,20($s2)`  | `Memoria[$s2+20]=$s1`<br>`$s1=0` oppure 1 | Memorizzazione di una parola come seconda fase di un’operazione atomica |
| Caricamento costante nella mezza parola superiore | `lui $s1,20`      | `$s1 = 20 * 2^16`                         | Caricamento di una costante nei 16 bit più significativi                |

### Logiche

| Istruzioni                    | Esempio           | Significato           | Commenti                                                            |
| ----------------------------- | ----------------- | --------------------- | ------------------------------------------------------------------- |
| And                           | `and $s1,$s2,$s3` | `$s1 = $s2 & $s3`     | Operandi in tre registri; AND bit a bit                             |
| Or                            | `or $s1,$s2,$s3`  | `$s1 = $s2 \| $s3`    | Operandi in tre registri; OR bit a bit                              |
| Nor                           | `nor $s1,$s2,$s3` | `$s1 = ~($s2 \| $s3)` | Operandi in tre registri; NOR bit a bit                             |
| And immediato                 | `andi $s1,$s2,20` | `$s1 = $s2 & 20`      | AND bit a bit tra un operando in registro e una costante            |
| Or immediato                  | `ori $s1,$s2,20`  | `$s1 = $s2 \| 20`     | OR bit a bit tra un operando in registro e una costante             |
| Scorrimento logico a sinistra | `sll $s1,$s2,10`  | `$s1 = $s2 << 10`     | Spostamento a sinistra del numero di bit specificato della costante |
| Scorrimento logico a destra   | `srl $s1,$s2,10`  | `$s1 = $s2 >> 10`     | Spostamento a destra del numero di bit specificato di una costante  |

### Salti condizionali

| Istruzioni                                         | Esempio            | Significato                                   | Commenti                                                           |
| -------------------------------------------------- | ------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| Salta se uguale                                    | `beq $s1,$s2,25`   | Se `($s1==$s2)` vai a PC+4+100                | Test di uguaglianza; salto relativo al PC                          |
| Salta se non è uguale                              | `bne $s1,$s2,25`   | Se `($s1!=$s2)` vai a PC+4+100                | Test di disuguaglianza; salto relativo al PC                       |
| Salta se ≤ 0                                       | `blez $s1,c`       | Se `($s1<=0)` vai all’etichetta `C`           | Comparazione di minoranza; salto relativo al PC                    |
| Salta se < 0                                       | `bltz $s1,c`       | Se `($s1<0)` vai all’etichetta `C`            | Comparazione di minoranza; salto relativo al PC                    |
| Salta se ≥ 0                                       | `bgez $s1,c`       | Se `($s1>=0)` vai all’etichetta `C`           | Comparazione di maggioranza; salto relativo al PC                  |
| Salta se ≥ 0                                       | `bgtz $s1,c`<br>   | Se `($s1>0)` vai all’etichetta `C`            | Comparazione di maggioranza; salto relativo al PC                  |
| Poni uguale a 1 se minore                          | `slt $s1,$s2,$s3`  | Se `($s2 < $s3) $s1 = 1` altrimenti `$s1 = 0` | Comparazione di minoranza; utilizzata con bne e beq                |
| Poni uguale a 1 se minore, numeri senza segno      | `sltu $s1,$s2,$s3` | Se `($s2 < $s3) $s1 = 1` altrimenti `$s1 = 0` | Comparazione di minoranza su numeri senza segno                    |
| Poni uguale a 1 se minore, immediato               | `slti $s1,$s2,20`  | Se `($s2 < 20) $s1 = 1` altrimenti `$s1 = 0`  | Comparazione di minoranza su una costante                          |

### Salti incondizionati

| Istruzioni           | Esempio    | Significato                          | Commenti                                                                                                      |
| -------------------- | ---------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Salto incondizionato | `j 2500`   | Vai a 10000                          | Salto all’indirizzo della costante                                                                            |
| Salto indiretto      | `jr $ra`   | Vai all’indirizzo contenuto in `$ra` | Salto all’indirizzo contenuto nel registro, utilizzato per il ritorno da procedura e per i costrutti *switch* |
| Salta e collega      | `jal 2500` | `$ra` = PC+4; vai a 10000            | Chiamata a procedura                                                                                          |


---
Created: 2024-03-06
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduzione
L’architettura MIPS (**M**icroprocessor without **I**nterlocked **P**ipelined **S**tages) è un'architettura per microprocessori RISC. Il disegno dell'architettura e del set di istruzioni è semplice e lineare, spesso utilizzato come caso di studio nei corsi universitari indirizzati allo studio delle architetture dei processori.
La memoria MIPS è indicizzata al byte. Dunque se mi trovo all’indirizzo t e devo leggere la parola successiva incremento indirizzo come t+4, questo perché una parola sono 4 byte ossia 32 bit (1 byte = 8 bit)

---
## Operandi MIPS

| Nome                       | Esempio                                                                               | Commenti                                                                                                                                                                                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 32 registri                | `$s0-$s7`,`$t0-$t9`, `$zero`, `$a0-$a3`, `$v0-$v1`, `$gp`, `$fp`, `$sp`, `$ra`, `$at` | Accesso veloce ai dati. Nel MIPS gli operandi devono essere contenuti nei registri per potere eseguire delle operazioni. Il registro `$zero` contiene sempre il valore 0, e il registro `$at` viene riservato all’assemblatore per la gestione di costanti molto lunghe                              |
| $2^{30}$ parole in memoria | `Memoria[0]`, `Memoria[4]`, ….                                                        | Alla memoria si accede solamente attraverso le istruzioni di trasferimento dati. Il MIPS utilizza l’**indirizzamento al byte**, perciò due parole consecutive hanno indirizzi in memoria a una distanza di 4. La memoria consente di memorizzare strutture dati, vettori o il contenuto dei registri |

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

### Salti incodizionati
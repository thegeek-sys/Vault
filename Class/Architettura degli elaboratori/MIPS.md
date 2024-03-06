---
Created: 2024-03-06
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## 

## Operandi MIPS

| Nome                       | Esempio                                                                               | Commenti                                                                                                                                                                                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 32 registri                | `$s0-$s7`,`$t0-$t9`, `$zero`, `$a0-$a3`, `$v0-$v1`, `$gp`, `$fp`, `$sp`, `$ra`, `$at` | Accesso veloce ai dati. Nel MIPS gli operandi devono essere contenuti nei registri per potere eseguire delle operazioni. Il registro `$zero` contiene sempre il valore 0, e il registro `$at` viene riservato all’assemblatore per la gestione di costanti molto lunghe                              |
| $2^{30}$ parole in memoria | `Memoria[0]`, `Memoria[4]`, ….                                                        | Alla memoria si accede solamente attraverso le istruzioni di trasferimento dati. Il MIPS utilizza l’**indirizzamento al byte**, perciò due parole consecutive hanno indirizzi in memoria a una distanza di 4. La memoria consente di memorizzare strutture dati, vettori o il contenuto dei registri |

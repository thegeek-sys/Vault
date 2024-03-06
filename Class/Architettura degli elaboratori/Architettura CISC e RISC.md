---
Created: 2024-03-06
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

## Differenze

| CISC                                                                                                                                                              | RISC                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Istruzioni di dimensioni variabile**<br>- Per il fetch della successiva è necessaria la decodifica della precedente                                             | **Istruzioni di dimensione fissa**<br>- Fetch della successiva senza decodifica della precedente                                                |
| **Formato variabile**<br>- Decodifica complessa                                                                                                                   | **Istruzioni di formato uniforme**<br>- Per semplificare la fase di decodifica                                                                  |
| **Operandi in memoria**<br>- Molti accessi alla memoria per istruzione                                                                                            | **Operazioni ALU solo tra registri**<br>- Senza accesso a memoria                                                                               |
| **Pochi registri interni**<br>- Maggior numero di accessi in memoria                                                                                              | **Molti registri interni**<br>- Per i risultati parziali senza accessi alla memoria                                                             |
| **Modi di indirizzamento complessi**<br>- Maggior numero di accessi in memoria<br>- Durata variabile dell’istruzione<br>- Conflitti tra istruzioni più complicati | **Modi di indirizzamento semplici**<br>- Con spiazzamento (un solo accesso a memoria)<br>- Durata fissa dell’istruzione<br>- Conflitti semplici |
| **Istruzioni complesse**<br>- Pipeline più complicata<br> - Più veloci a svolgere operazioni complesse                                                            | **Istruzioni semplici**<br>- Pipeline più veloce<br>- Più lento nello svolgere operazioni complesse                                             |

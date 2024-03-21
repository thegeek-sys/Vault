---
Created: 2024-03-14
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Compilatore|Compilatore]]
>- [[#Assemblatore|Assemblatore]]
>- [[#Linker|Linker]]

---

![[Screenshot 2024-03-14 alle 21.26.04.png]]
## Compilatore
Il compilatore ci permette di trasformare codice alto livello in **codice Assembly**
In particolare:
- istruzioni/espressioni di alto livello → gruppi di istruzioni ASM
- variabili temporanee → registri
- variabili globali e locali → etichette e direttive
- strutture di controllo → salti ed etichette
- funzioni e chiamate → etichette e salti a funzione
- chiamate a funzioni esterne → tabella per linker

## Assemblatore
L’assemblatore ci permette di trasformare il codice assembly in **codice oggetto**
In particolare:
- istruzioni ASM → istruzioni macchina
- etichette → indirizzi o offset relativi
- direttive → allocazione e inizializzazione strutture statiche
- macro → gruppi di istruzioni

## Linker
Il linker definisce la posizione in memoria delle strutture dati statiche e del codice.
“Collega” i riferimenti a:
- chiamate di funzioni esterne → salti non relativi
- strutture dati esterne → indirizzamenti non relativi
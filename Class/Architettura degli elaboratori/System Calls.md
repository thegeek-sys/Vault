---
Created: 2024-03-21
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#Richieste al sistema operativo|Richieste al sistema operativo]]
>- [[#Hello World!|Hello World!]]

---
## Introduction
![[Screenshot 2024-03-19 alle 12.40.53.png|center|400]]
Una system call è un’istruzione speciale che trasferisce il controllo dalla modalità utente alla modalità kernel, viene per questo utilizzato per richiedere un servizio a livello kernel del sistema operativo del computer in uso.
Questo viene effettuato attraverso un’istruzione speciale detta *TRAP* che in Assembly corrisponde all’istruzione `syscall`; infatti il codice relativo ai servizi del sistema operativo è eseguibile soltanto in kernel mode per ragioni di sicurezza.
Una volta terminato il compito relativo alla particolare chiamata di sistema invocata, il contraolla ritorna al processo chiamante passando dal kernel mode allo user mode.

---
## Richieste al sistema operativo
Input:
- `$v0` → operazione richiesta
- `$a0...$a2 , $f0` → eventuali parametri
Output:
- `$v0, $f0` → eventuale risultato

| Descrizione    | Syscall<br>($v0) | Argomenti                                  | Risultato        |
| :------------- | :--------------: | :----------------------------------------- | :--------------- |
| Stampa intero  |        1         | $a0 intero                                 |                  |
| Stampa float   |        2         | $f12 float                                 |                  |
| Stampa double  |        3         | $f12 float                                 |                  |
| Stampa stringa |        4         | $a0 string address                         |                  |
| Leggi intero   |        5         |                                            | intero (in $v0)  |
| Leggi float    |        6         |                                            | float (in $v0)   |
| Leggi double   |        7         |                                            | double (in $v0)  |
| Leggi stringa  |        8         | \$a0 = buffer address<br>\$a1 = num chars. |                  |
| sbrk           |        9         | $a0 amount                                 | address (in $v0) |
| Fine programma |        10        |                                            |                  |

---
## Hello World!

```arm-asm
.globl main

.data
string: .asciiz "Hello world!"

.text
main:
li $v0,4
la $a0,string

syscall
```


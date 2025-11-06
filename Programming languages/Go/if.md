---
Programming language: "[[Programming languages/Go/Go|Go]]"
Related:
---
---
## Regole e scope
Come per il `for`, anche nell’`if` non è necessaria nessuna parentesi `()` per le condizioni ma sono sempre necessarie le graffe `{}`.

Un `if` in Go ha la particolarità che può iniziare con un’istruzione eseguita prima della condizione (*short statement*). Inoltre per quanto riguarda lo scoping, le variabili dichiarate nello short statement sono visibili solo all’interno del blocco `if` e dei successivi `else`.

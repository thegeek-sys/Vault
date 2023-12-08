---
Created: 2023-11-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Line-profiler ci serve per controllare il tempo di esecuzione di una determinata funzione. Per renderlo efficacie ci basta aggiungere prima della funzione il decoratore (un profile pattern che potenzia la funzione) `@profile` (se la macchina non ha installato line-profiler avrò un errore e dovrò fare una exception).

---
## Installation
`conda install spyder-line-profiler -c conda-forge`
[Guida d’uso](https://docs.spyder-ide.org/current/plugins/lineprofiler.html)

---
## Exception
Queste tre righe di codice mi permettono di non ricevere errori quando non eseguiamo il profiler ma comunque sono presenti gli `@profile`

```python
import builtins
if 'profile' not in dir(builtins):
    def profile(X) : return X
```

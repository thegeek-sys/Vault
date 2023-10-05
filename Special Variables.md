---
Created: 2023-10-05
Programming language: "[[Python]]"
Related:
  - "[[Modules]]"
Completed:
---
---
## `__file__`
Questa variabile contiene il percorso del file corrente nel filesystem dal punto in cui viene invocato

---
## `__path__`
E’ utile per determinare il percorso da dove l’interprete Python cerca i moduli all’interno del programma

---
## `__name__`
In base a come viene eseguito il modulo vale:
- il nome del modulo (che coincide con il nome del file) se usato con `import`
- la stringa `"__main__"` se il modulo e' eseguito come uno script.
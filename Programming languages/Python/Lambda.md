---
Created: 2023-10-30
Programming language: "[[Python]]"
Related:
  - "[[Functions]]"
Completed:
---
---
## Introduction
Le **funzioni anonime** o **lambda function** sono funzioni che posso definire "al volo" dovergli dare un nome (per questo anonime). Queste sono spesso usate come chiave `key` nell'ordinamento per specificare ordinamenti parziali.
Definisce una funzione al volo che prende in **ingresso x, y** e ritorna il valore generato dall'espressione. Sintassi generale:
```python
lambda x: <espressione che usa x> 
#un unica riga e una sola espressione
```

Se a `sorted` passiamo `key=lambda elmen: (elem)` genera una tupla che è usata per ordinare i valori
```python
L  = [
    #nome     cognome    voto    eta
    ('mario'  ,'rossi'  , 23  , 24),
    ('mario'  ,'ferro'  , 30  , 21),
    ('mario'  ,'rossi'  , 19  , 32),
    ('frank' ,'bianc'   , 27  , 24),
    ('marta'  ,'veri'   , 25  , 29),
    ('mario'  ,'rossi'  , 21  , 32),
]
#: list[tuple[str, str, int, int]]

S = sorted(L, reverse=False, key=lambda elem: elem)
# ('frank', 'bianc', 27, 24)     Frank è prima di tutti
# ('mario', 'ferro', 30, 21)     A parità di nome ordina il cognome
# ('mario', 'rossi', 19, 32)     A parità di cognome ordina voto
# ('mario', 'rossi', 21, 32)  
# ('mario', 'rossi', 23, 24)  
# ('marta', 'veri', 25, 29)

S = sorted(L, reverse=False, key=lambda elem:
						   (elem[1],  # prima per cognome crescente
			                elem[0],  # poi per nome
				            -elem[3],  # poi per età al contrario
                            elem[2]) )# infine per voto crescente
# ('frank', 'bianc', 27, 24)  
# ('mario', 'ferro', 30, 21)  
# ('mario', 'rossi', 19, 32)  
# ('mario', 'rossi', 21, 32)  
# ('mario', 'rossi', 23, 24)  
# ('marta', 'veri', 25, 29)
```
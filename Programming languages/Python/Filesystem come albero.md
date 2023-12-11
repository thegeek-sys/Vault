---
Created: 2023-12-11
Programming language: "[[Python]]"
Related:
  - "[[Ricorsione]]"
Completed:
---
---
## Introduction
Una struttura come potrebbe essere quella del **filesystem** è facilmente analizzabile in modo ricorsivo. Creiamo dunque una funzione che data una directory (o un nodo) lista tutti i file e le sotto directory.

> [!WARNING]
> Spesso all’esame viene chiesto di esplorare una cartella in modo ricorsivo

---
# Esercizio tipo

 > Mi viene dato un percorso assoluto e devo ritornare un lista contente tutti i file all’interno del path con una determinata estensione `ext`

```python
def find_file_with_ext(folder, ext):
	rez = [] # init vuoto
	for fname in os.listdir(folder):
		# mi rende i file e dir in folder
		# mi riprocuro il percorso assoluto
		full_path = folder+'/'+fname
		if os.path.isfile(full_path):
			# se siamo nel caso del file controllo estensione
			if full_path.endswith(ext):
				# ok, lo aggiungo alla lista
				rez.append(full_path)
		else:
			# ottengo lista di file delle sotto-directory
			L_files = find_file_with_ext(full_path, ext)
			# unisco i file correnti con sottodir
			rez = rez + L_files
			# rez.extend(L_files)
	# torno per eventuali chiamate sopra di me
	return rez
```

---

>  Mi viene dato un percorso assoluto e devo ritornare un dizionario contente come `key` tutti i file all’interno del path con una determinata estensione  e come `value` la dimensione in byte
>  `file: size_byte`

```python
def find_file_with_ext(folder, ext):
	rez = {}
	for fname in os.listdir(folder):
		full_path = folder+'/'+fname
		if os.path.isfile(full_path):
			if full_path.endswith(ext):
				rez[full_path] = os.stat(full_path).st_size
		else:
			D_files = find_file_with_ext(full_path, ext)
			rez.update(D_files)
	return rez
```
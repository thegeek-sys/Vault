---
Created: 2023-11-21
Programming language: "[[Python]]"
Related:
  - "[[Matrix]]"
  - "[[Shallow and Deep Copy]]"
---
---
## Introduction
Le immagini possono essere:
- **raster** → griglia che quantizza i colori (formati: `jpeg`, `png`, `tiff`)
	- Possono essere rappresentate come:
		- Scala di grigi → lista di liste di valore in scala di grigi
		- RGB → lista di liste di tuple che indicano i colori RGB
- **vettoriali** → curve e tracciati matematici che descrivono l’immagine (formati: `svg`, `eps`, `pdf`)

In realtà le immagini RGB più formalmente possono essere viste come:
- un tensore di dimensioni `HxWx3`
- una matrice di profondità 3
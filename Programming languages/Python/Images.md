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


![[matrix rgb.png|150]]
In realtà le immagini RGB più formalmente possono essere viste come:
- un tensore di dimensioni `HxWx3`
- una matrice di profondità 3

![[sistema riferimento rgb.png|300]]
Il loro sistema di riferimento è ordinato da sinistra a destra e dall’alto al basso (lo 0,0 si trova in alto a sinistra)

---
##  Disegnare su immagini

```python
colormap = {'red': (255,0,0), 'blu':(0,0,255), 'green':(0,255,0),
			'black':(0,0,0), 'white':(255,255,255)}
# mi è comodo per definire dei colori standard
color_mat = create_matrix_lc(256,256, colormap['black'])

'''
N.B. ricordarsi  che  mat è indicizata [riga][colonna] oppure [y][x]
'''

# per scrivere su una riga per x+lenght
def plot_line_h(mat, x, y, lenght, value):
	# selezioniamo la riga ci scriviamo sopra da x a x+lenght
	mat[y][x:x+lenght] = [value] * lenght

# per scrivere su più righe lungo tutto x
def plot_line_w(mat, x, y, lenght, value):
	# ci muoviamo da  y fino a y+lenght , x rimane fissa
	for each_y in range(y,y+lenght):
		mat[each_y][x] = value
```
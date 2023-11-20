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

### Linee

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
	# ci muoviamo da y fino a y+lenght, x rimane fissa
	for each_y in range(y,y+lenght):
		mat[each_y][x] = value
```

### Rettangoli

```python
def  plot_rect(mat, x, y, w, h, value, clip=False):
	'''
	plottiamo il rettangolo:
	1. upper segment
	2. lower segment
	3. left segment
	4. right segment
	'''
	del clip(v, min_v, max_v):
		return min(max(min_v, v), max_v)
	
	H = len(mat)
	W = len(mat[0])
	
	# per ridimensionare il rettangolo se va in overflow rispetto
	# ai margini dell'immagine
	if clip:
		x, y = clip(x, 0,  W-1), clip(y, 0, H-1)
		w, h = clip(w, 0,  W-1-x), clip(h, 0, H-1-y)

	# plotting
	plot_line_h(mat, x,     y,     w, value) # 1.
	plot_line_h(mat, x,     y+h-1, w, value) # 2.
	plot_line_v(mat, x,     y,     h, value) # 3.
	plot_line_v(mat, x+w-1, y,     h, value) # 4.

# un secondo modo per evitare di sbordare
def draw_pixel2(img, x, y, colore):
	altezza = len(img)
	larghezza = len(img[0])
	if 0 <= x < larghezza and 0 <= y < altezza:
		img[y][x] = colore
```

---
## Aprire immagini

```python
import images
im = images.load('gopttp_small.png')

# scrivere riga nera al centro dell'immagine
R = len(im)
W = len(im[0])
im[H//2] = [(0,)*3]*W
images.save(im, 'giotto_edited.png')

images.visd('giotto_edited.png') # mi permette di fare un render direttamente in sypder su iphyton

```
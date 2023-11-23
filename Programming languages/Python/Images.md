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
Installazione librerie
```python
get_ipython().system(' wget https://twiki.di.uniroma1.it/pub/Programmazione1/AA20_21/DiarioDelleLezioni-CanaleAL/png.py.txt &>/dev/null')
get_ipython().system(' mv png.py.txt png.py')
import png
import io 

# from Prof. Andrea Sterbini
class Image:                                                                                                                    
    '''Oggetto che contiene una immagine come lista di liste di colori (R,G,B) e che viene                                         
    direttamente visualizzate in IPython console/qtconsole/notebook col metodo _repr_png_'''  
    
    def __init__(self, img, mode='RGB'):                                                                                                       
        self.pixels = img  
        self.mode = mode

    def _repr_png_(self):                                                                                                          
        '''Produce la rappresentazione binaria della immagine in formato PNG'''                                                    
        if self.pixels:
            img = png.from_array(self.pixels, self.mode)                                                                                   
            b = io.BytesIO()                                                                                                           
            img.save(b)                                                                                                                
            return b.getvalue()

get_ipython().system('wget https://twiki.di.uniroma1.it/pub/Programmazione1/AA20_21/DiarioDelleLezioni-CanaleAL/images.py.txt &>/dev/null')
get_ipython().system(' mv images.py.txt images.py')
import images
```

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
def  plot_rect(mat, x, y, Wr, Hr, value, clip=False):
	'''
	plottiamo il rettangolo:
	1. upper segment
	2. lower segment
	3. left segment
	4. right segment
	
	# x,y -------------- x+Wr-1,y
    # |                     |
    # |                     |                                          
    # |                     |                     
    # |                     |                     
    # x,y+Hr-1,---------x+Wr-1,y+Hr-1
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
	plot_line_h(mat, x,      y,      Wr, value) # 1.
	plot_line_h(mat, x,      y+Hr-1, Wr, value) # 2.
	plot_line_v(mat, x,      y,      Hr, value) # 3.
	plot_line_v(mat, x+Wr-1, y,      Hr, value) # 4.

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

---
## Flippare immagini
### Asse verticale

```python
def flip_v(img):
	H = len(img)
	W = len(img[0])
	flipped_img = []
	for each_y in range(H):
		# sto invertendo ogni colonna
		flipped_img.append(img[each_y][::-1])
	return flipped_img

def flip_v_map(img):
	H = len(img)
	W = len(img[0])
	flipped = map(lambda each_r: each_r[::-1], img)
	return  list(flipped)
```

### Asse orizzontale

```python
def flip_h(img):
	H = len(img)
	W = len(img[0])
	flipped_img = []
	for each_y_flip in reversed(range(H)): # range(H-1,-1,-1)
		# sto invertendo ogni riga
		flipped_img = [row[::-1] for row in im]
	return flipped_img

def flip_h_map(img):
	H = len(img)
	W = len(img[0])
	return list(map(lambda each_rr: each_rr, reversed(img)))
```

---
## Shape di una matrice
```python
def shape(mat):
    # immediatly check empty matrix
    if len(mat) == 0:
        return 0, 0
    if len(mat[0]) == 0:
        return 1, 0
    # rows corresponds to height
    r = len(mat)
    c = len(mat[0])
    return r, c
```
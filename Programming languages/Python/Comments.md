---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: 
Completed: false
---
---
## Index
- [Introduction](#Introduction)
- [Line Comment](#Line%20Comment)
- [Multiple Lines Comments](#Multiple%20Lines%20Comments)
---
## Introduction 
Comments are parts of code that are not executed by the compiler

---
## Line Comment

Comments starts with a `#`, and Python will ignore them:

```Python
#This is a comment  
print("Hello, World!")
```

---
Comments can be placed at the end of a line, and Python will ignore the rest of the line:

``` Python
print("Hello, World!") #This is a comment
```

---
## Multiple Lines Comments
- Python does not really have a syntax for multiline comments.
- Since Python will ignore string literals that are not assigned to a variable, you can add a multiline string (triple quotes) in your code, and place your comment inside it:

```Python
"""  
This is a comment  
written in  
more than just one line  
"""  
print("Hello, World!")
```

---
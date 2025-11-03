---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction

>[!quote] NISTRIR 7298
>The process of granting or denying specific requests to:
>1. obtain and use information and related information processing sercives
>2. enter specific physical facilities

>[!quote] RFC 4949
>A process by which use of system resources is regulated according to a security policy and is permitted only by authorized entities (users, programs, processes, or other systems) according to that policy

### Access control concepts
![[Pasted image 20251103223231.png]]

### Access control models
#### Discretionary access control (DAC)
Controls access based on the identity of the requestor and on access rules (authorizations) stating what requestors are (or are not) allowed to do.
#### Mandatory access control (MAC)
Controls access based on comparing security labels with security clearances.
#### Role-based access control (RBAC)
Controls access based on the roles that users have within the system and on rules stating what accesses are allowed to users in given roles.
#### Attribute-based access control (ABAC)
Controls access based on attributes of the user, the resource to be accessed, and current environmental conditions

---
## Subjects, objects and access rights
### Subject
A subject is an entity capable of accessing objects. There are three classes of subject:
- owner
- group
- world

### Object
An object is a resource to which access is controlled and is used to contain and/or receive information

### Access right
Describes the way in which a subject may access an object. It could include:
- read
- write
- execute
- delete
- create
- search

---
## Discretionary Access Control (DAC)
**Discretionary Access Control** (*DAC*) is a scheme in which an entity may be granted access rights that permit the entity, by its own volition, to enable another entity to access some resource. It is often provided using an **access matrix**

>[!example] Example of access control matrix
>![[Pasted image 20251103223908.png]]
>
>One dimension describes identified subjects asking data access to the resources while the other lists the objects that may be accessed. Each entry of the matrix indicates the access rights of a particular subject for a particular object and an empty cell means that no access rights are granted.

### Access control list
It defines, for each object a list called object’s access control list, which enumerates all the subjects that have access right for the objects and, for each such subject, gives the access rights that the subject has for the object

![[Pasted image 20251103224315.png]]

>[!info] Capabilities
>- takes a subject-centered approach to access controls
>- defines, for each subject $s$, the list of the object for which $s$ has nonempty access control rights, together with a specific rights for each such object

#### Extended access control matrix
The extended version considers the ability of one subject to transfer rights, create another subject and to have `owner` access right to that subject.

![[Pasted image 20251103224554.png|600]]

It can also define a hierarchy of subjects

| Rule | Command (by $S_0$)                                                    | Authorization                                        | Operation                                                                           |
| ---- | --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| R1   | transfer $\begin{Bmatrix} \alpha^* \\ \alpha \end{Bmatrix}$ to $S, X$ | “$\alpha^*$” in $A[S_0, X]$                          | store $\begin{Bmatrix} \alpha^* \\ \alpha \end{Bmatrix}$ in $A[S, X]$               |
| R2   | grant $\begin{Bmatrix} \alpha^* \\ \alpha \end{Bmatrix}$ to $S, X$    | ‘owner’ in $A[S_0, X]$                               | store $\begin{Bmatrix} \alpha^* \\ \alpha \end{Bmatrix}$ in $A[S, X]$               |
| R3   | delete $\alpha$ from $S, X$                                           | ‘control’ in $A[S_0, S]$  or  ‘owner’ in $A[S_0, X]$ | delete $\alpha$ from $A[S, X]$                                                      |
| R4   | $w \leftarrow \text{read } S, X$                                      | ‘control’ in $A[S_0, S]$  or  ‘owner’ in $A[S_0, X]$ | copy $A[S, X]$ into $w$                                                             |
| R5   | create object $X$                                                     | None                                                 | add column for $X$ to $A$; store ‘owner’ in $A[S_0, X]$                             |
| R6   | destroy object $X$                                                    | ‘owner’ in $A[S_0, X]$                               | delete column for $X$ from $A$                                                      |
| R7   | create subject $S$                                                    | None                                                 | add row for $S$ to $A$; execute **create object $S$**; store ‘control’ in $A[S, S]$ |
| R8   | destroy subject $S$                                                   | ’owner’ in $A[S_0, S]$                               | delete row for $S$ from $A$; execute **destroy object $S$**                         |

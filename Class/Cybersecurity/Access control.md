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

### Organization of the access control function
Every access by a subject to an object is mediates by the controller for that object and the controller’s decision is based on the current contents of the matrix.
Certain subjects have the authority to make specific changes to the access matrix. 

![[Pasted image 20251103225011.png|500]]

>[!example] Unix subjects, objects, rights
>- subjects → users, groups, others
>- objects → files, directories
>- access rights: read, write, execute
>	- for files
>		- read → reading from a file
>		- write → writing to a file
>		- execute → executing a (program) file
>	- for directories
>		- read → list the files within the directory
>		- write → create, rename, or delete files within the directory
>		- execute → enter the directory

### UNIX file access control
Every user has a unique user identification number and each member of a primary group is identifies by a group ID.

There are 12 protection bits, specifying read, write, and execute permission for the owner of the file, members of the group and all other users.

>[!info]
>The owner ID, group ID and protection bits are part of the file’s inode

#### Traditional UNIX File Access Control (minimal ACL)
In UNIX systems, file access control mechanisms are essential for maintaining system security and managing user permissions. The traditional model, based on minimal Access Control Lists (ACLs), uses a combination of permission bits and special attributes to control how users and groups interact with files and directories. The main components of this model include SetUID, SetGID, the Sticky Bit, and the Superuser privilege.

##### Set User ID (SetUID) and Set Group ID (SetGID)
The SetUID and SetGID mechanisms allow the system to temporarily use the rights of the file owner or group in addition to those of the real user when making access control decisions. This mechanism enables certain privileged programs to access files or resources that are not generally accessible to regular users, allowing controlled elevation of privileges when necessary.
##### Sticky Bit
The Sticky Bit is another important attribute, typically applied to directories. When a directory has the Sticky Bit set, only the owner of a file within that directory is allowed to rename, move, or delete the file, even if other users have write permissions for the directory.
##### Superuser
the Superuser is exempt from the usual access control restrictions. The superuser account has complete system-wide access, allowing it to read, modify, and execute any file or process on the system. This level of privilege is necessary for system administration tasks but must be handled carefully to maintain system security and stability.

#### Access Control Lists (ACLs) in UNIX
Many modern UNIX systems support access control lists (FreeBSD, OpenBSD, Linux, Solaris, …). In FreeBSD:
- there is `setfacl` command to assigns a list of UNIX user IDs and groups
- any number of users and groups can be associated with a file
- there are read, write, execute protection bits
- a file does not need to have an ACL
- is included an additional protection bit that indicates whether the file has an extended ACL

When a process requests access to a file system object two steps are performed:
1. selects the most appropriate ACL
2. checks if the matching entry contains sufficient permissions

#### Extended access control list
![[Pasted image 20251103230539.png|400]]

---
## Mandatory Access Control (MAC)
It is inspired by the Bell-La Padula model in which each subject and each object is assigned to a security class.

In the simples formulation, security classes from a strict hierarchy and are referred to as **security levels**. A subject is said to have a *security clearance* of a given levels and an object is said to have a *security classification* of a given level

The security classes control the manner by  which a subject may access an object.

>[!bug] Limitations
>- cannot manage the “downgrade of objects”
>- classification creep

### Multilevel security (MLS)
The model defined four access modes:
- **read** → the subject is allowed only read access to the object
– **append** → the subject is allowed only write access to the object
– **write** → the subject is allowed both read and write access to the object
– **execute** → the subject is allowed neither read nor write access to the object but may invoke the object for execution

Confidentiality is achieved if a subject at a high level may not convey information to a subject at a lower level unless that flow accurately reflects the will of an authorized user as revealed by an authorized declassification.

#### Multilevel security confidentiality
MLS has the following characteristics:

- no read up → a subject can only read an object of less or equal security level. This is referred to in the literature as the simple security property (*ss-property*).
- no write down: A subject can only write into an object of greater or equal security level. This is referred to in the literature as the \*-property (pronounced starproperty).

The more recent MAC implementations, are SELinux and AppArmor for
Linux and Mandatory Integrity Control for Windows

---
## Role-based Access Control
In RBAC, rather than specifying access control rights for subjects directly, you define roles and then specify access control for these roles

![[Pasted image 20251104095553.png]]

The goal is to describe organizational access control policies based on job function (a user’s permissions are determined by her roles rather than by identity or clearance).
This helps increasing flexibility/scalability in policy administration. In particular it helps:
- meeting new security requirements
- reduce errors in administration
- reduce cost of administration

![[Pasted image 20251104095858.png|450]]

To sum up:
- roles are defined based on job function → e.g. bookkeeper
- permissions are defines based on job authority and responsibilities within a role → e.g. bookkeeper is allowed to read financial records
- users have access to object based on the assigned role → e.g. Sally is the bookkeeper

### Access control matrix representation of RBAC
![[Pasted image 20251104100150.png]]

>[!example] Exercise
>Given the following User Assignment and Permission Assignment define the corresponding access matrix
>
>![[Pasted image 20251104101044.png]]
>
>>[!done] Solution
>>
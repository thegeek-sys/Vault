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
>>[!done]- Solution
>>![[Pasted image 20251104101231.png|400]]

### Family of Role-Based Access Control models
![[Pasted image 20251104101309.png|400]]

### RBAC1: role hierarchy
Some roles subsume others. This hierarchy is used when many operations are common to a large number of roles and reflect an organization’s role structure.

In this way instead of specifying permissions for each role, one specifies it for a more generalized role, so that granting access to role $R$ implies that access is granted for all specialized roles of $R$.

#### Role hierarchy
Structuring roles, partial order ≤:
$$
x\leq y\quad\text{we say }x\text{ is specialization of }y
$$
Inheritance of permission from generalized role $y$ (top) to specialized role $x$ (bottom):
- members of $x$ are also implicitly members of $y$
- if $x\leq y$ then role $x$ inherits permissions of role $y$

Partial order:
- reflexivity → $x\leq x$
- transitivity → $x\leq y \land y\leq x\to x\leq z$
- antisymmetry → $x\leq y\land y\leq x\to x=y$

>[!example]
>![[Pasted image 20251104102144.png|450]]

>[!example] Exercise
>$$UA=\{(u_{1},r_{2}),(u_{2},r_{3}),(u_{3},r_{4}),(u_{4},r_{5})\}$$
>$$PA=\{(r_{1},p_{1}),(r_{2},p_{2}),(r_{3},p_{3}),(r_{4},p_{4}),(r_{5},p_{5})\}$$
>
>Given the following role hierarchy, determine the permissions that users have in form of an access matrix
>
>>[!done]- Solution
>>![[Pasted image 20251104102426.png|400]]

### RBAC2: constrains
RBAC2 provides a means of adapting RBAC to specifics of administrative and security policies of an organization and a defined relationship among roles or a condition related to roles.

Types:
- *mutually exclusive roles* → a user can only be assigned to one role in the set (either during a session or statically) and any permission (access right) can  be granted to only one role in the set
- *cardinality* → setting a maximum number with respect to roles
- *prerequisite roles* → dictates that a user can only be assigned to a particular role if it is already assigned to some other specified role

---
## Attribute-Based Access Control (ABAC)
Attribute-Based Access Control can define authorizations that express conditions on properties of both the resource and the subject.
Its strength is its flexibility and expressive power, but its main obstacle to its adoption in real systems had been concern about the performance impact of evaluating predicates on both resource and user properties for each access.

Web services have been pioneering technologies through the introduction of the *eXtensible Access Control Markup Language* (**XAMCL**) and there is considerable interest in applying the model to cloud services

There are many kinds of attributes:
- **subject attributes**
	- a subject is a an active entity that causes information to flow among objects or changes the system state
	- attributes define the identity and characteristics of the subject
- **object attributes**
	- an object (or resource) is a passive information system-related entity containing or receiving information
	- objects have attributes that can be leverages to make access control decisions
- **environment attributes**
	- describe the operational, technical, and even situational environment or context in which the information access occurs
	- these attributes have so far been largely ignored in most access control policies

![[Pasted image 20251104104717.png|400]]

ABAC is distinguishable because it controls access to objects by evaluating rules against the attributes of entities, operations, and the environment relevant to a request. It relies upon the evaluation of attributes of the subject, attributes of the object, and a formal relationship or access control rule defining the allowable operations for subject-object attribute combinations in a given environment

ABAC systems are capable of enforcing DAC, RBAC, and MAC concepts as it allows an unlimited number of attributes to be combined to satisfy any access control rule

### Policies
A policy is a set of rules and relationships that govern allowable behavior within an organization, based on the privileges of subjects and how resources or objects  are to be protected under which environment conditions (typically written from the perspective of the object that needs protecting and the privileges available to subjects).

Privileges represents the authorized behavior of a subject and are defined by an authority and embodies in a policy.

>[!info] Policies model
>- $S$, $O$ and $E$ are subjects, objects and environments, respectively
>- $SA_{k}(1,\dots ,k, \dots, K)$, $OA_{m}(1,\dots m, \dots, M)$ and $EA_{n}(1,\dots,n, \dots N)$ are the pre-defined attributes for subjects, objects, and environments, respectively
>- $ATTR(s)$, $ATTR(o)$, $ATTR(e)$ are attributes assignement relations, for example
>	- $role(s)=\text{"Service Consumer"}$
>	- $ServiceOwner(o)=\text{"XYZ, Inc"}$
>	- $CurrentDate(e)=\text{"01-23-2005"}$
>- rule → $\text{can\_access}(s,o,e)\leftarrow f(\text{ATTR}(s), \text{ATTR}(o), \text{ATTR}(e))$

>[!example]
>$$\begin{align}R_{1}:\text{can\_access}&(u,m,e)\leftarrow \\ \\&(\text{Age}(u)\leq 17 \land \text{Rating}(m)\in \{R, PG-13, G\}) \lor \\&(\text{Age}(u)\geq 13 \land \text{Age}(u)<17 \land \text{Rating}(m)\in \{PG-13,G\}) \lor \\&(\text{Age}(u)<13\land \text{Rating}(m)\in G)\end{align}$$
>
>![[Pasted image 20251104110010.png|400]]

### ABAC vs RBAC
In RBAC as the number of attributes increases to accommodate finer-grained policies, the number of roles and permission grows exponentially

$$
\prod^K_{k=1}\text{Range}(SA_{k})\land \prod^M_{m=1}\text{Range}(SA_{m})
$$

The ABAC model deals with additional attributes in an efficient way

>[!example] Finer grained policy example
>Movies are classified as either New release or Old Release, based on release date compared to the current date
>Users are classified as Premium User and Regular User, based on the fee they pay
>
>Policy → only premium users can view new movies
>
>In RBAC you need to double the number of roles, to distinguish each user by age and fee and to double the number of separate permissions
>
>| Roles            | Permissions      |
>| ---------------- | ---------------- |
>| Adult-Regular    | R-Old_release    |
>| Juvenile-Regular | PG13-Old_release |
>| Child-Regular    | G-Old_release    |
>| Adult-Premium    | R-New_release    |
>| Juvenile-Premium | PG13-New_release | 
>
>
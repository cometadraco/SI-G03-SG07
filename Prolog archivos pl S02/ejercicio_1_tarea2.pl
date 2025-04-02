padre(carlos, juan).
padre(juan, pedro).
padre(juan, lucia).
padre(jorge, ana).
padre(carlos, jose).
padre(jose, roberto).
padre(jose, claudia).

madre(maria, juan).
madre(ana, pedro).
madre(ana, lucia).
madre(elena, ana).
madre(maria, jose).
madre(luisa, roberto).
madre(luisa, claudia).

hermano(pedro, lucia).
hermano(roberto, claudia).
hermano(jose, juan).
hermano(juan, jose).

hermana(lucia, pedro).
hermana(claudia, roberto).

hijo(X,Y) :- padre(Y,X).
hijo(X,Y) :- madre(Y,X).

abuelo(X,Y) :- padre(X,Z), padre(Z,Y).
abuelo(X,Y) :- padre(X,Z), madre(Z,Y).
abuelo(X,Y) :- madre(X,Z), padre(Z,Y).
abuelo(X,Y) :- madre(X,Z), madre(Z,Y).

tia(X,Y) :- hermana(X,Z), padre(Z,Y).
tia(X,Y) :- hermana(X,Z), madre(Z,Y).

tio(X,Y) :- hermano(X,Z), padre(Z,Y).
tio(X,Y) :- hermano(X,Z), madre(Z,Y).














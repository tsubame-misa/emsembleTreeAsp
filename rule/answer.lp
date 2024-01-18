% Define

class(K) :- predict_class(X, K).

% this will eliminate rules that apply to less than 10 instances
invalid(X) :- rule(X), size(X,S), S < 6.
valid(X) :- rule(X), not invalid(X).


% X is dominated by Y if ...
ge_f1_leq_size_geq_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy), Fx < Fy, Sx >= Sy, Spx <= Spy.
geq_f1_le_size_geq_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy),Fx <= Fy, Sx > Sy, Spx <= Spy.
geq_f1_leq_size_ge_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy),Fx <= Fy, Sx >= Sy, Spx < Spy.

dominated :- valid(Y), ge_f1_leq_size_geq_sup(Y).
dominated :- valid(Y), geq_f1_le_size_geq_sup(Y).
dominated :- valid(Y), geq_f1_leq_size_ge_sup(Y).

% cannot be dominated
:- not dominated.

% total number of conditions should not exceed 30
:- #sum { S,X : size(X,S), selected(X) } > 80.


% maximize accuracy and support, minimize the number of conditions
#maximize { A,X : selected(X), accuracy(X,A)}.
#maximize { A,X : selected(X), f1_score(X,A)}.
%#maximize { S,X : selected(X), support(X,S)}.
%#minimize { L,X : selected(X), size(X,L)}.



% number of shared conditions between rules
rule_overlap(X,Y,Cn) :- selected(X), selected(Y), X!=Y,
Cn = #count { Cx : Cx=Cy, condition(X,Cx), condition(Y,Cy) }.
#minimize { Cn,X : selected(X), selected(Y), rule_overlap(X,Y,Cn) }.


% action
1{selected(X) : predict_class(X, K), valid(X)} 10 :- class(K).


% Display
#show selected/1.
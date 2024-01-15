
%rule(X).            %X holds the rule index.
%condition(X,I).     %Rule X has condition I.
%support(X,S).       %Support S of rule X, the number of instances that is covered by rule X.
%size(X,L).          % Number of conditions in rule X (length, L).
%error_rate(X,E).    %Error rate (1 − accuracy), E, of the rule X evaluated in the training data.
%accuracy(X,A).      %Accuracy score of rule X.
%precision(X,P).     %Precision score of rule X.
%recall(X,R).        %Recall score of rule X.
%f1_score(X,F).      %F1-score of rule X.
%predict_class(X,C). %Predicted class label C of rule 


% ルール

rule(1). condition(1,1). condition(1,2). condition(1,3). support(1,10).
size(1,3). accuracy(1,50). error_rate(1,50). precision(1,50).
recall(1,50). f1_score(1,50). predict_class(1,1).

rule(3). condition(3,1). condition(3,2). condition(3,3). support(3,5).
size(3,3). accuracy(3,50). error_rate(3,50). precision(3,30).
recall(3,40). f1_score(3,30). predict_class(3,1).

rule(2). condition(2,1). condition(2,2). condition(2,3). support(2,10).
size(1,3). accuracy(2,30). error_rate(2,50). precision(2,40).
recall(2,10). f1_score(2,40). predict_class(2,1).

rule(4). condition(4,1). condition(4,2). condition(4,3). support(4,10).
size(4,3). accuracy(4,40). error_rate(4,30). precision(4,70).
recall(4,30). f1_score(4, 60). predict_class(4,1).


% Define

class(K) :- predict_class(X, K).

valid(X) :- rule(X), not invalid(X).
% this will eliminate rules that apply to less than 10 instances
invalid(X) :- rule(X), support(X,S), S < 10.


% X is dominated by Y if ...
ge_f1_leq_size_geq_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy), Fx < Fy, Sx >= Sy, Spx <= Spy.
geq_f1_le_size_geq_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy),Fx <= Fy, Sx > Sy, Spx <= Spy.
geq_f1_leq_size_ge_sup(Y) :- selected(X), valid(Y), size(X,Sx), size(Y,Sy), f1_score(X,Fx), f1_score(Y,Fy), support(X,Spx), support(Y,Spy),Fx <= Fy, Sx >= Sy, Spx < Spy.

dominated :- valid(Y), ge_f1_leq_size_geq_sup(Y).
dominated :- valid(Y), geq_f1_le_size_geq_sup(Y).
dominated :- valid(Y), geq_f1_leq_size_ge_sup(Y).

% cannot be dominated
:- dominated.

% total number of conditions should not exceed 30
:- #sum { S,X : size(X,S), selected(X) } >= 6.

% maximize accuracy and support, minimize the number of conditions
#maximize { A,X : selected(X), accuracy(X,A)}.
#maximize { S,X : selected(X), support(X,S)}.
#minimize { L,X : selected(X), size(X,L)}.

% number of shared conditions between rules
rule_overlap(X,Y,Cn) :- selected(X), selected(Y), X!=Y,
Cn = #count { Cx : Cx=Cy, condition(X,Cx), condition(Y,Cy) }.
#minimize { Cn,X : selected(X), selected(Y), rule_overlap(X,Y,Cn) }.


% action
1 { selected(X) : predict_class(X, K), valid(X) } 2 :- class(K).




% Display
#show selected/1.
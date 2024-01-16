% Define

class(K) :- predict_class(X, K).

valid(X) :- rule(X), not invalid(X).
% this will eliminate rules that apply to less than 10 instances
invalid(X) :- rule(X), support(X,S), S < 20.


% action
1{selected(X) : predict_class(X, K)} 10 :- class(K).


% Display
#show selected/1.
#lang racket
(provide parse parse-define parse-e)
(require "ast.rkt")

;; [Listof S-Expr] -> Prog
(define (parse s)
  (match s
    [(cons (and (cons 'define _) d) s)
     (match (parse s)
       [(Prog ds e)
        (Prog (cons (parse-define d) ds) e)])]
    [(cons e '()) (Prog '() (parse-e e))]
    [_ (error "program parse error")]))

;; S-Expr -> Defn
; (define (parse-define s)
;   (match s
;     [(list 'define (list-rest (? symbol? f) xs) e)
;      (if (andmap symbol? xs)
;          (Defn f xs (parse-e e))
;          (error "parse definition error"))]
;     [_ (error "Parse defn error" s)]))

;; S-Expr -> Defn
(define (parse-define s)
  (match s 
    [(list 'define (? symbol? f)
           (list-rest 'case-lambda cs))
     (Defn f (FunCase (parse-case-lambda-clauses cs)))]   
    [(list 'define (cons (? symbol? f) xs) e)
     (if (all symbol? xs)
         (Defn f (parse-param-list xs e))
         (error "parse definition error"))]
    [_ (error "Parse defn error" s)]))

;; like andmap, but work on improper lists too
(define (all p? xs)
  (match xs
    ['() #t]
    [(cons x xs) (and (p? x) (all p? xs))]
    [x (p? x)]))
    
;; S-Expr -> [Listof FunCaseClause]
(define (parse-case-lambda-clauses cs)
  (match cs
    ['() '()]
    [(cons c cs)
     (cons (parse-case-lambda-clause c)
           (parse-case-lambda-clauses cs))]
     [_
      (error "parse case-lambda error")]))

;; S-Expr -> FunRest
(define (parse-case-lambda-clause c)
  (match c
    [(list (? symbol? x) e)
     (FunRest '() x (parse-e e))]
    [(list xs e)
     (parse-param-list xs e)]))    

;; S-Expr S-Expr -> FunPlain or FunRest
(define (parse-param-list xs e)
  (match xs
    ['() (FunPlain '() (parse-e e))]
    [(cons x xs)
     (match (parse-param-list xs e)
       [(FunPlain xs e) (FunPlain (cons x xs) e)]
       [(FunRest xs y e) (FunRest (cons x xs) y e)])]
    [(? symbol? xs)
     (FunRest '() xs (parse-e e))]
    [_
     (error "parse parameter list error")]))

;; S-Expr -> Expr
(define (parse-e s)
  (match s
    [(? integer?)                  (Int s)]
    [(? boolean?)                  (Bool s)]
    [(? char?)                     (Char s)]
    [(? string?)                   (Str s)]
    ['eof                          (Eof)]
    [(? symbol?)                   (Var s)]
    [(list 'quote (list))          (Empty)]
    [(list (? (op? op0) p0))       (Prim0 p0)]
    [(list (? (op? op1) p1) e)     (Prim1 p1 (parse-e e))]
    [(list (? (op? op2) p2) e1 e2) (Prim2 p2 (parse-e e1) (parse-e e2))]
    [(list (? (op? op3) p3) e1 e2 e3)
     (Prim3 p3 (parse-e e1) (parse-e e2) (parse-e e3))]


    [(cons (? (op? opn) p) es)      (PrimN p (map parse-e es))] 


    [(list 'begin e1 e2)
     (Begin (parse-e e1) (parse-e e2))]
    [(list 'if e1 e2 e3)
     (If (parse-e e1) (parse-e e2) (parse-e e3))]
    [(cons 'cond cs)           (parse-cond cs)]
    [(cons 'case (cons ev cs)) (parse-case ev cs)]
    [(list 'let  bs e)      (parse-let  bs e)]
    [(list 'let* bs e)      (parse-let* bs e)]
    [(list-rest 'apply (? symbol? f) es) (parse-apply f es)]
    [(cons 'match (cons e ms))
     (parse-match (parse-e e) ms)]
    [(cons 'values es) (Values (parse-values es))]
    [(list 'let-values (list (list xs e)) e0) 
     (LetValues xs (parse-e e) (parse-e e0))]
    [(list (or 'lambda 'λ) xs e)
     (if (and (list? xs)
              (andmap symbol? xs))
         (Lam (gensym 'lambda) xs (parse-e e))
         (error "parse lambda error"))]
    [(cons e es)
     (App (parse-e e) (map parse-e es))]    
    [_ (error "Parse error" s)]))

(define (parse-values es)
 (match es
   ['() '()]
   [(list e t ...) (cons (parse-e e) (parse-values t))]))

;; Id S-Expr -> Expr
(define (parse-apply f es)
  (match es
    [(list e) (Apply f '() (parse-e e))]
    [(cons e es)
     (match (parse-apply f es)
       [(Apply f es e0)
        (Apply f (cons (parse-e e) es) e0)])]
    [_ (error "parse apply error")]))
    
;; S-Expr S-Expr -> Let
(define (parse-let bs e)
  (match bs
    ['() (Let '() '() (parse-e e))]
    [(cons (list (? symbol? x1) e1) bs)
     (match (parse-let bs e)
       [(Let xs es e)
        (Let (cons x1 xs) (cons (parse-e e1) es) e)])]
    [else (error "parse error")]))

;; S-Expr S-Expr -> Let
(define (parse-let* bs e)
  (match bs
    ['() (Let* '() '() (parse-e e))]
    [(cons (list (? symbol? x1) e1) bs)
     (match (parse-let* bs e)
       [(Let* xs es e)
        (Let* (cons x1 xs) (cons (parse-e e1) es) e)])]
    [else (error "parse error")]))



;; S-Expr -> Cond
(define (parse-cond cs)
  (match cs
    [(list (list 'else e)) (Cond '() (parse-e e))]
    [(cons (list p e) css)
     (match (parse-cond css)
       [(Cond cs el)
        (Cond (cons (Clause (parse-e p) (parse-e e)) cs) el)])]
    [_ (error "parse error")]))    

;; S-Expr S-Expr -> Case
(define (parse-case ev cs)
  (match cs
    [(list (list 'else e)) (Case (parse-e ev) '() (parse-e e))]
    [(cons (list ds e) css)
     (match (parse-case ev css)
       [(Case ev cs el)
        (Case ev (cons (Clause (parse-datums ds) (parse-e e)) cs) el)])]
    [_ (error "parse error")]))



(define (parse-match e ms)
  (match ms
    ['() (Match e '() '())]
    [(cons (list p r) ms)
     (match (parse-match e ms)
       [(Match e ps es)
        (Match e
               (cons (parse-pat p) ps)
               (cons (parse-e r) es))])]))

(define (parse-pat p)
  (match p
    [(? boolean?) (PLit p)]
    [(? integer?) (PLit p)]
    [(? char?)    (PLit p)]
    ['_           (PWild)]
    [(? symbol?)  (PVar p)]
    [(list 'quote (list))
     (PLit '())]
    [(list 'box p)
     (PBox (parse-pat p))]
    [(list 'cons p1 p2)
     (PCons (parse-pat p1) (parse-pat p2))]
    [(list 'and p1 p2)
     (PAnd (parse-pat p1) (parse-pat p2))]))

(define op0
  '(read-byte peek-byte void))

(define op1
  '(add1 sub1 zero? char? write-byte eof-object?
         integer->char char->integer
         box unbox empty? cons? box? car cdr
         vector? vector-length string? string-length
         - abs integer? boolean? not))
(define op2
  '( < = cons eq? make-vector vector-ref make-string string-ref arithmetic-shift))
(define op3
  '(vector-set!))

(define opn 
  '(+ -))


(define (op? ops)
  (λ (x)
    (and (symbol? x)
         (memq x ops))))
;; S-Expr -> [Listof Datum]
(define (parse-datums ds)
  (match ds
    ['() '()]
    [(cons (? integer? i) ds)
     (cons i (parse-datums ds))]
    [(cons (? boolean? b) ds)
     (cons b (parse-datums ds))]
    [(cons (? char? c) ds)
     (cons c (parse-datums ds))]
    [(cons 'eof ds)
     (cons eof (parse-datums ds))]
    [_ (error "parse error")]))
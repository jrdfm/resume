# Multiple return values

**[Overview](#overview)**<br>
**[AST and Parser](#ast-and-parser)**<br>
**[Interpreter](#interpreter)**<br>
**[Compiler](#compiler)**<br>
**[Runtime](#runtime)**<br>
**[Epilogue](#epilogue)**<br>

## Overview

The objective of this project is to implement values and let-values; *values* will take a list of n expressions and return all their values, *let-values* will evaluate the first expression, which should produce values equal to the number of variables, binds them and then evaluates the body. 
```racket
(values e1 ... en)

(let-values ([(x1 ... xn) e]) e0)
```
For convenience I included the implementation on both iniquity-plus and loot, which we will walkthrough now.


## AST and Parser
First we need to update the ast with the following two additions,

```racket
...
(struct Values (es)        #:prefab)
(struct LetValues (xs e e0) #:prefab)
...
```
The parser is the most straightforward part of this project. For values we will use a helper function, parse-values , that parses each expression and returns a list, and for let-values we'll just parse e and e0.
```racket
...
[(cons 'values bs) (Values (parse-values bs))]

[(list 'let-values (list (list xs e)) e0) 
    (LetValues xs (parse-e e) (parse-e e0))]
...
```
## Interpreter
The interpreter was surpassingly tricky, maybe even more than the compiler. For values the obvious thing to do was to use interp-env* to interpret the list of expressions but then I ran into the problem of what to do with the list of values it returns, namely how to return them all. The solution was to use apply and values, which took me a while to figure out. 

let-values was even more tricker. We need to interpret e, which most likely will return values and we have the problem of capturing those values and binding them to xs. Luckily racket has a function call-with-values which takes two procedures, a generator and a receiver. The generator expression is expected to produce multiple values we can use to call the receiver. So we'll interpret e in the body of a λ expression that takes no arguments, the generator, and use the list function as a receiver, and bind the list it returns to i, then interpret e0 in the extended environment. This again took some trial and error. 
 

```racket
...
[(Values es) (apply values (interp-env* es r ds))]

[(LetValues xs e e0) 
    (let ((i (call-with-values (λ () (interp-env e r ds)) list)))
    (interp-env e0 (append (map list xs i) r) ds))]))
...
```
Now that's out of the way we can move onto the fun stuff.    
## Compiler

### Compiler for values
Values takes a list of arbitrary number of expressions, including 0, so my approach was to use compile-es and communicate result arity via a dedicated register, r11. But we need to make the distinction between cases where es has only one argument and all the other cases. 

In the one expression case we'll just use compile-e, move one to r11 and the result will be in rax, which means all contexts which expect one value will function as usual.To avoid code duplication we'll use the function assert-res-arity through out the compiler, which will take the expected number of results and jumps to 'raise_error_align if isn't equal to r11. Note that we need to assert even the single expression compiles to a single value because values takes in a list of expressions that each resolve to a single value, i.e. we can't have (values (values 1 2)).    

In all other cases we'll use compile-es which will push all the results on the stack, then move the length of the list of expressions to r11. But we're not quite done yet, what if values is the top-level expression? We can't just leave the returns on the stack for the runtime because we need to pop all our pushes, otherwise it'll be a segfault. 

The solution was to use the heap. To do that we'll define a new type val in types.rkt with value #b111, and it'll be just like a vector with its length upfront. So after compiling the expressions we'll call push-to-heap which takes n as an argument and moves n values off the top of the stack. Then returns a pointer tagged with type value, which we can just pass around through rax. That makes things easier because now values just returns one value via rax regardless of the number of arguments it took. So we can have values as the top-level expression without an issue. We now just have to deal with the only other context where values is appropriate , let-values.

```racket

(define (compile-values es c)
(let ((i (length es)))
  (if (= 1 i)   ;; if (values e) -> compile e
      (seq (compile-e (car es) c #f)
           (assert-res-arity 1)
           (Mov r11 1))
      (seq
       (compile-es es c)
       (Mov r11 i)
       (push-to-heap i)))))

```
### Compiler for let-values
let-values should compile e, which should produce (length xs) values and evaluate the body with the variables bound to the values. So after compiling e we need to check if r11 is equal to the number of variables and then check if it produces values, with a values pointer in rax, or if its just a regular return. In the first case we'll remove the pointer tag and call the function push-to-stack, which takes an offset , i and the heap pointer through rax then moves i values to the stack. In the later case we'll just push rax. Then compile e0 in xs appended to the compile time environment, and pop the stack at the end. 
```racket
(define (compile-let-values xs e e0 c)
  (let ((nv (gensym 'nv))
        (f (gensym 'f) )
        (i (length xs)))
    (seq (compile-e e c #f) ;; check if val and push from heap to stack
         (Cmp r11 i) ;; check res arity
         (Jne 'raise_error_align)
         (Mov r9 rax)
         (And r9 ptr-mask)
         (Cmp r9 type-val)
         (Jne nv)
         (Xor rax type-val)
         (push-to-stack 0 i) ;; push values to stack
         (Jmp f)
         (Label nv)
         (Push rax)
         (Label f)
         (compile-e e0 (append xs c) #f);; either 1 via rax or val through push stack
         (Add rsp (* 8 i)))))
```
### Everything else
We're now left with the less fun task of going through each compile-e expression throughout the compiler and adding result arity checking according to the context. Which was't that bad because compile-prim1, compile-prim2 and compile-prim3 all take expressions that return one value. So the only adjustment to compile-ops is in read-byte and peek-byte, just move one to r11, we'll also need to do this after compiling immediate values , variables and strings. 

We're just left with if , match, let and begin. If's predicate need to be a single value, but the then-expr and else-expr can return any amount of values. Match seems complicated but we only need to assert result arity of the expression we're matching, and if we find a match the body can return any number of values.

For let the first expression need to resolve to a singular value but the body can return multiple values, and in begin both expressions can return multiple values, we'll just ignore e1 as usual.  

## Runtime
My approach to the runtime for this project was similar to my approach to it throughout the semester, to ignore it unless I have to deal with it. And it needs to be dealt with now. 

First order of business is to add changes to types.rkt to types.h, then add a type T_VAL to the typedef of type_t. We then need to create a struct much like val_vect_t, we'll call it val_val_t, and add changes to Wrap/unwrap values. 

In order to print values appropriately print.c should be adjusted too, so we'll add a case for T_VAL in print_result function, which will call print_result_interior. And in print_result_interior we'll add a case for T_VAL that will call print_val. 

```c
void print_val(val_val_t *v)
{
  uint64_t i;
  if (!v) return; 
  for (i = 0; i < v->len; ++i) {
    print_result_interior(v->elems[(v->len -1) - i]);
    if (i < v->len - 1)
     putchar ('\n');
  }
}

```
As it can be seen above it looks very much like print_vect except for the weird indexing ,we need to print in reverse, and the newline at the end. 
I also added the following code to unload-value in unload-bits-asm.rkt to use asm-interp for testing and debugging. 

```racket
...
[(? val-bits? i)
    (if (zero? (untag i))
        (void)
        (for-each (λ (x) (writeln x)) 
        (reverse (build-list (heap-ref i)
                    (λ  (j)
                        (unload-value (heap-ref (+ i (* 8 (add1 j))))))))))]
...
```

## Epilogue

What's next? I plan to add features throughout the assignments and have a version that'll showcase everything we did. I also want to add some form of static type checking. Despite my fear of the runtime I'm glad I got to work on it, because it turned out I completely misunderstood some of it. Working on this project and trying to articulate what I did has been fun. Thank you for your time.  
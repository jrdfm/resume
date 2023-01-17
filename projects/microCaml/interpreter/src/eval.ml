open MicroCamlTypes
open Utils

exception TypeError of string
exception DeclareError of string
exception DivByZeroError 

(* Provided functions - DO NOT MODIFY *)

(* Adds mapping [x:v] to environment [env] *)
let extend env x v = (x, ref v)::env

(* Returns [v] if [x:v] is a mapping in [env]; uses the
   most recent if multiple mappings for [x] are present *)
let rec lookup env x =
  match env with
  | [] -> raise (DeclareError ("Unbound variable " ^ x))
  | (var, value)::t -> if x = var then !value else lookup t x

(* Creates a placeholder mapping for [x] in [env]; needed
   for handling recursive definitions *)
let extend_tmp env x = (x, ref (Int 0))::env

(* Updates the (most recent) mapping in [env] for [x] to [v] *)
let rec update env x v =
  match env with
  | [] -> raise (DeclareError ("Unbound variable " ^ x))
  | (var, value)::t -> if x = var then (value := v) else update t x v
        
(* Part 1: Evaluating expressions *)

(* Evaluates MicroCaml expression [e] in environment [env],
   returning a value, or throwing an exception on error *)


let eval_art o e e' = 
let x,y = match e, e' with 
| Int i,Int j -> i, j
| _  -> raise (TypeError "Expected type int") in 
match o with 
| Add -> (x + y)
| Sub -> (x - y)
| Mult -> (x * y)
| Div -> if y <> 0 then x/y else raise DivByZeroError
| _ -> raise (TypeError "Unexpected operator")
  

let eval_rel o e e' = 
let x,y = match e, e' with 
| Int i,Int j -> i, j
| _  -> raise (TypeError "Expected type int") in 
match o with 
| Greater -> (x > y)
| Less -> (x < y)
| GreaterEqual -> (x >= y)
| LessEqual -> (x <= y)
| _ -> raise (TypeError "Unexpected operator")

let eval_con o e e' = 
  let x,y = match e, e' with 
  | String i,String j -> i, j
  | _  -> raise (TypeError "Expected type string") in 
   x^y 

let eval_eq o e e' = 
  let x,y = match e, e' with 
          | Int i,Int j -> Int i ,Int j 
          | String i,String j -> String i ,String j
          | Bool i, Bool j ->  Bool i,Bool j
          | _  -> raise (TypeError "Cannot compare types")  in
  match o with 
| Equal ->  x = y
| NotEqual -> x <> y
| _ -> raise (TypeError "Unexpected operator")

  
let eval_bl o e e' = 
  let x,y = match e, e' with
          | Bool i, Bool j ->  i,j
          | _  -> raise (TypeError "Expected type bool")  in
  match o with 
| And ->  x && y
| Or -> x || y
| _ -> raise (TypeError "Unexpected operator")  




let rec eval_expr env e =  match e with 
| ID x -> lookup env x
| Value v -> v
| Fun (x,e) -> Closure (env,x,e)
| Not e -> (match eval_expr env e  with 
          | Bool b ->  Bool (not b)   
          | _ ->  raise (TypeError "Expected type bool")
             )
| Binop  (op , ex1 , ex2) -> let e1 = eval_expr env ex1 in 
                             let e2 = eval_expr env ex2 in 
                            (match op with 
                            | Add | Sub | Mult | Div-> Int (eval_art op e1 e2)
                            | Greater | Less | GreaterEqual | LessEqual -> Bool (eval_rel op e1 e2) 
                            | Concat -> String (eval_con op e1 e2 )
                            | Equal | NotEqual -> Bool (eval_eq op e1 e2 )
                            | And | Or -> Bool (eval_bl op e1 e2)
                            )               
| If (e1, e2,e3) -> ( match eval_expr env e1 with 
                      | Bool true -> (eval_expr env e2)
                      | Bool false -> (eval_expr env e3)
                      | _ -> raise (TypeError "Expected type bool")
                     )

| FunctionCall  (e1,e2) -> (match (eval_expr env e1) with 
                           | Closure (cen,cv,cex) ->  let v1 = eval_expr env e2 in
                                                      (eval_expr (extend cen cv v1) cex)
                           | _ -> raise (TypeError "Not a function")
                            )
                           
| Let (id,rc ,e1 ,e2) -> (match rc with 
                              | false -> let v1 = eval_expr env e1 in 
                                         let env' = extend env  id v1 in 
                                         let v2 = eval_expr env' e2 in
                                         v2
                              | true -> let env' = extend_tmp env id in
                                        let v1 = eval_expr env' e1 in 
                                        update env' id v1 ;
                                        eval_expr env' e2
                          )




(* Part 2: Evaluating mutop directive *)

(* Evaluates MicroCaml mutop directive [m] in environment [env],
   returning a possibly updated environment paired with
   a value option; throws an exception on error *)
let eval_mutop env m =  match m with 
| Def (id,ex)-> let env' = extend_tmp env id in
                let v1 = eval_expr env' ex in 
                update env' id v1 ;
                (env',Some v1)
| Expr ex ->  let v = eval_expr env ex in
               (env,Some v)
| NoOp -> ([],None)

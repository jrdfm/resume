
(*  
S -> M + S | M
M -> n
Where n is any integer   
*)



Expr -> LetExpr | IfExpr | FunctionExpr | OrExpr
LetExpr -> let Recursion Tok_ID = Expr in Expr
    Recursion -> rec | ε
FunctionExpr -> fun Tok_ID -> Expr
IfExpr -> if Expr then Expr else Expr
OrExpr -> AndExpr || OrExpr | AndExpr
AndExpr -> EqualityExpr && AndExpr | EqualityExpr
EqualityExpr -> RelationalExpr EqualityOperator EqualityExpr | RelationalExpr
    EqualityOperator -> = | <>
RelationalExpr -> AdditiveExpr RelationalOperator RelationalExpr | AdditiveExpr
    RelationalOperator -> < | > | <= | >=
AdditiveExpr -> MultiplicativeExpr AdditiveOperator AdditiveExpr | MultiplicativeExpr
    AdditiveOperator -> + | -
MultiplicativeExpr -> ConcatExpr MultiplicativeOperator MultiplicativeExpr | ConcatExpr
    MultiplicativeOperator -> * | /
ConcatExpr -> UnaryExpr ^ ConcatExpr | UnaryExpr
UnaryExpr -> not UnaryExpr | FunctionCallExpr
FunctionCallExpr -> PrimaryExpr PrimaryExpr | PrimaryExpr
PrimaryExpr -> Tok_Int | Tok_Bool | Tok_String | Tok_ID | ( Expr )

let lookahead tok_list = match tok_list with
        [] -> raise (IllegalExpression "lookahead")
        | (h::t) -> h

let match_tok tok tok_list =
  match tok_list with
    | (h::t) when tok = h -> t
    | _ -> raise (IllegalExpression "match_tok")


let rec parse_expr toks = 
    let (rem_toks, expr) = parse_Let toks in
and parse_S toks =
    let (rem_toks, expr) = parse_If toks in
      match (lookahead rem_toks) with
      | Tok_Add -> let toks2 = match_token rem_toks Tok_Add in
            let (toks3, expr2) = parse_S toks2 in
            (toks3, Plus (expr, expr2))
      | _ -> (rem_toks, expr)





let rec parse_expr toks = parse_Exp toks
and parse_Exp toks = 
    if lookahead toks = Tok_Let then parse_Let toks 
    
    else raise (InvalidInputException "parser") 
and parse_Let t = 
    (*  let tt,ee = parse_Fun t in   *)
    let t = match_token toks Tok_Let in
    let t',e = parse_Rec t in 
    (*if lookahead t' = Tok_ID then*)
      let id = lookahead t' in
      let tt = match_token t' Tok_ID in
      let tt',ex1 = parse_expr tt in 
      let tt''= match_token tt' Tok_In in
      let ttt',ex2 = parse_expr tt'' in 

      (ttt',Let (id,e,ex1,ex2))
and parse_Rec t = match t with
    | Tok_Rec -> let t' = match_token t Tok_Rec in 
                (t',true)
    | _ -> (t,false) 



  (* parse_Exp toks
and parse_Exp = 
    if lookahead toks = Tok_Let then parse_Let toks 
    else if lookahead toks = Tok_If then parse_If toks 
    else if lookahead toks = Tok_Fun then parse_Fun toks 
    else if lookahead toks = Tok_Or then parse_Or toks 
    else raise (InvalidInputException "parser") 
and parse_Let t = 
    (*  let tt,ee = parse_Fun t in   *)
    let t = match_token toks Tok_Let in
    let t',e = parse_Rec t in 
    (*if lookahead t' = Tok_ID then*)
      let id = lookahead t' in
      let t'' = match_token t' Tok_ID in
      let t''',ex1 = parse_expr t'' in 
      let t''''= match_token t''' Tok_In 
      let t''''',ex2 = parse_expr t'''' in 

      (t''''',Let (id,e,ex1,ex2))
    
and parse_Rec t = match t with
    | Tok_Rec -> let t' = match_token t Tok_Rec in 
                (t',true)
    | _ -> (t,false) 
(* IfExpr -> if Expr then Expr else Expr  *)
and parse_If t = let tt = match_token toks Tok_If in
                 let tt',ex1 = parse_expr tt in 
                 let tt'' = match_token tt' Tok_Then in
                 let tt''',ex2 = parse_expr tt'' in 
                 let ttt = match_token tt''' Tok_Else in
                 let ttt',ex3 = parse_expr ttt in 
                 If(ex1,ex2,ex3)

and parse_Fun t = let tt,e = parse_If t in 
                 (* if lookahead t = Tok_ID then *)
                 let id = lookahead t' in
                 let t'' = match_token t' Tok_ID in
                 let t''',ex = parse_expr t'' in 

                 Fun(id,ex)
                 


                  






  




  *)
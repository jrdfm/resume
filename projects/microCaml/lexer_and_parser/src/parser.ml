open MicroCamlTypes
open Utils
open TokenTypes

(* Provided functions - DO NOT MODIFY *)

(* Matches the next token in the list, throwing an error if it doesn't match the given token *)
let match_token (toks : token list) (tok : token) =
  match toks with
  | [] -> raise (InvalidInputException (string_of_token tok))
  | h :: t when h = tok -> t
  | h :: _ ->
      raise
        (InvalidInputException
           (Printf.sprintf "Expected %s from input %s, got %s"
              (string_of_token tok)
              (string_of_list string_of_token toks)
              (string_of_token h)))

(* Matches a sequence of tokens given as the second list in the order in which they appear, throwing an error if they don't match *)
let match_many (toks : token list) (to_match : token list) =
  List.fold_left match_token toks to_match

(* Return the next token in the token list as an option *)
let lookahead (toks : token list) =
  match toks with [] -> None | h :: t -> Some h

(* Return the token at the nth index in the token list as an option*)
let rec lookahead_many (toks : token list) (n : int) =
  match (toks, n) with
  | h :: _, 0 -> Some h
  | _ :: t, n when n > 0 -> lookahead_many t (n - 1)
  | _ -> None

(* Part 2: Parsing expressions *)

let get_str t =
  match t with
  | [] -> raise (InvalidInputException "Empty Token List")
  | h :: t -> (
      match h with
      | Tok_String i | Tok_ID i -> i
      | _ -> raise (InvalidInputException "No matching token"))

let get_int t =
  match t with
  | [] -> raise (InvalidInputException "Empty Token List")
  | h :: t -> (
      match h with
      | Tok_Int i -> i
      | _ -> raise (InvalidInputException "No matching token"))

let get_bl t =
  match t with
  | [] -> raise (InvalidInputException "Empty Token List")
  | h :: t -> (
      match h with
      | Tok_Bool i -> i
      | _ -> raise (InvalidInputException "No matching token"))

let rec parse_expr toks = parse_Exp toks

and parse_Exp toks =
  match lookahead toks with
  | Some Tok_Let -> parse_Let toks
  | Some Tok_If -> parse_If toks
  | Some Tok_Fun -> parse_Fun toks
  | _ -> parse_Or toks

and parse_Let toks =
  let t = match_token toks Tok_Let in
  let t', rec_ex = parse_Rec t in

  let id = get_str t' in
  let tt = match_token t' (Tok_ID id) in
  let tt' = match_token tt Tok_Equal in

  let tt'', ex1 = parse_expr tt' in
  let ttt' = match_token tt'' Tok_In in
  let ttt'', ex2 = parse_expr ttt' in

  (ttt'', Let (id, rec_ex, ex1, ex2))

and parse_Rec t =
  match lookahead t with
  | Some Tok_Rec -> (match_token t Tok_Rec, true)
  | _ -> (t, false)

and parse_If t =
  let tt = match_token t Tok_If in
  let tt', ex1 = parse_expr tt in
  let tt'' = match_token tt' Tok_Then in
  let tt''', ex2 = parse_expr tt'' in
  let ttt = match_token tt''' Tok_Else in
  let ttt', ex3 = parse_expr ttt in
  (ttt', If (ex1, ex2, ex3))

and parse_Fun t =
  let tt = match_token t Tok_Fun in

  let id = get_str tt in
  let tt' = match_token tt (Tok_ID id) in
  let tt'' = match_token tt' Tok_Arrow in
  let ttt'', ex = parse_expr tt'' in
  (ttt'', Fun (id, ex))

and parse_Or t =
  let t', and_ex = parse_And t in
  let look = lookahead t' in
  match look with
  | Some Tok_Or ->
      let tt = match_token t' Tok_Or in
      let tt', or_ex = parse_Or tt in
      (tt', Binop (Or, and_ex, or_ex))
  | _ -> (t', and_ex)

and parse_And t =
  let tt, eql_ex = parse_Eq t in
  match lookahead tt with
  | Some Tok_And ->
      let tt' = match_token tt Tok_And in
      let ttt', and_ex = parse_And tt' in
      (ttt', Binop (And, eql_ex, and_ex))
  | _ -> (tt, eql_ex)

and parse_Eq t =
  let tt, rel_ex = parse_Rel t in
  match lookahead tt with
  | Some Tok_Equal ->
      let tt' = match_token tt Tok_Equal in
      let ttt', eq_ex = parse_Eq tt' in
      (ttt', Binop (Equal, rel_ex, eq_ex))
  | Some Tok_NotEqual ->
      let tt' = match_token tt Tok_NotEqual in
      let ttt', eq_ex = parse_Eq tt' in
      (ttt', Binop (NotEqual, rel_ex, eq_ex))
  | _ -> (tt, rel_ex)

and parse_Rel t =
  let tt, add_ex = parse_Add t in
  match lookahead tt with
  | Some Tok_Less ->
      let t' = match_token tt Tok_Less in
      let t'', r = parse_Rel t' in
      (t'', Binop (Less, add_ex, r))
  | Some Tok_Greater ->
      let t' = match_token tt Tok_Greater in
      let t'', r = parse_Rel t' in
      (t'', Binop (Greater, add_ex, r))
  | Some Tok_LessEqual ->
      let t' = match_token tt Tok_LessEqual in
      let t'', r = parse_Rel t' in
      (t'', Binop (LessEqual, add_ex, r))
  | Some Tok_GreaterEqual ->
      let t' = match_token tt Tok_GreaterEqual in
      let t'', r = parse_Rel t' in
      (t'', Binop (GreaterEqual, add_ex, r))
  | _ -> (tt, add_ex)

and parse_Add t =
  let tt, mul_ex = parse_Mul t in
  match lookahead tt with
  | Some Tok_Add ->
      let t' = match_token tt Tok_Add in

      let tt', add_ex = parse_Add t' in
      (tt', Binop (Add, mul_ex, add_ex))
  | Some Tok_Sub ->
      let t' = match_token tt Tok_Sub in

      let tt', add_ex = parse_Add t' in
      (tt', Binop (Sub, mul_ex, add_ex))
  | _ -> (tt, mul_ex)

and parse_Mul t =
  let tt, con_ex = parse_Con t in
  match lookahead tt with
  | Some Tok_Mult ->
      let t' = match_token tt Tok_Mult in
      let tt', mul_ex = parse_Mul t' in
      (tt', Binop (Mult, con_ex, mul_ex))
  | Some Tok_Div ->
      let t' = match_token tt Tok_Div in
      let tt', mul_ex = parse_Mul t' in
      (tt', Binop (Div, con_ex, mul_ex))
  | _ -> (tt, con_ex)

and parse_Con t =
  let tt, un_ex = parse_Un t in
  match lookahead tt with
  | Some Tok_Concat ->
      let t' = match_token tt Tok_Concat in
      let tt', con_ex = parse_Con t' in
      (tt', Binop (Concat, un_ex, con_ex))
  | _ -> (tt, un_ex)

and parse_Un t =
  match lookahead t with
  | Some Tok_Not ->
      let t' = match_token t Tok_Not in
      let tt', un_ex = parse_Un t' in
      (tt', Not un_ex)
  | _ -> parse_Func t

and parse_Func t =
  let tt, pr_ex = parse_Pri t in
  match lookahead tt with
  | Some (Tok_Int i) ->
      let tt', pr2_ex = parse_Pri tt in
      (tt', FunctionCall (pr_ex, pr2_ex))
  | Some (Tok_Bool i) ->
      let tt', pr2_ex = parse_Pri tt in
      (tt', FunctionCall (pr_ex, pr2_ex))
  | Some (Tok_String i) ->
      let tt', pr2_ex = parse_Pri tt in
      (tt', FunctionCall (pr_ex, pr2_ex))
  | Some (Tok_ID i) ->
      let tt', pr2_ex = parse_Pri tt in
      (tt', FunctionCall (pr_ex, pr2_ex))
  | Some Tok_LParen ->
      let tt', pr2_ex = parse_Pri tt in
      (tt', FunctionCall (pr_ex, pr2_ex))
  | _ -> (tt, pr_ex)

and parse_Pri t =
  match lookahead t with
  | Some (Tok_Int i) ->
      let ex = Value (Int (get_int t)) in
      let tt = match_token t (Tok_Int i) in
      (tt, ex)
  | Some (Tok_Bool bl) ->
      let ex = Value (Bool bl) in
      let tt = match_token t (Tok_Bool bl) in
      (tt, ex)
  | Some (Tok_String str) ->
      let ex = Value (String (get_str t)) in
      let tt = match_token t (Tok_String str) in
      (tt, ex)
  | Some (Tok_ID str) ->
      let ex = ID (get_str t) in
      let tt = match_token t (Tok_ID str) in
      (tt, ex)
  | Some Tok_LParen ->
      let tt' = match_token t Tok_LParen in
      let tt'', exp = parse_expr tt' in
      let ttt' = match_token tt'' Tok_RParen in
      (ttt', exp)
  | _ -> raise (InvalidInputException "parse_pri")

(* Part 3: Parsing mutop *)

let rec parse_mutop toks =
  match lookahead toks with
  | Some Tok_DoubleSemi ->
      let tt = match_token toks Tok_DoubleSemi in
      (tt, NoOp)
  | Some Tok_Def -> parse_Def toks
  | _ ->
      let tt, e = parse_expr toks in
      let tt' = match_token tt Tok_DoubleSemi in
      (tt', Expr e)

and parse_Def t =
  let tt = match_token t Tok_Def in
  let id = get_str tt in
  let tt' = match_many tt [ Tok_ID id; Tok_Equal ] in
  let tt'', ex = parse_expr tt' in
  let ttt' = match_token tt'' Tok_DoubleSemi in
  (ttt', Def (id, ex))

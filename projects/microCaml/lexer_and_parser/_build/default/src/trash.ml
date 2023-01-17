
let rec parse_expr toks : expr_result = parse_E toks

and parse_E (toks : token list) : token list * expr = parse_O toks

and parse_O (toks : token list) : token list * expr =
  let t, a = parse_A toks in
  match lookahead t with
  | Tok_Or ->
      let t' = match_token t Tok_Or in
      let t'', o = parse_O t' in
      (t'', Or (a, o))
  | _ -> (t, a)

and parse_A (toks : token list) : token list * expr =
  let t, ee = parse_EE toks in
  match lookahead t with
  | Tok_And ->
      let t' = match_token t Tok_And in
      let t'', a = parse_A t' in
      (t'', And (ee, a))
  | _ -> (t, ee)

and parse_EE (toks : token list) : token list * expr =
  let t, r = parse_R toks in
  match lookahead t with
  | Tok_Equal ->
      let t' = match_token t Tok_Equal in
      let t'', ee = parse_EE t' in
      (t'', Equal (r, ee))
  | Tok_NotEqual ->
      let t' = match_token t Tok_NotEqual in
      let t'', ee = parse_EE t' in
      (t'', NotEqual (r, ee))
  | _ -> (t, r)

and parse_R (toks : token list) : token list * expr =
  let t, ae = parse_AE toks in
  match lookahead t with
  | Tok_Less ->
      let t' = match_token t Tok_Less in
      let t'', r = parse_R t' in
      (t'', Less (ae, r))
  | Tok_Greater ->
      let t' = match_token t Tok_Greater in
      let t'', r = parse_R t' in
      (t'', Greater (ae, r))
  | Tok_LessEqual ->
      let t' = match_token t Tok_LessEqual in
      let t'', r = parse_R t' in
      (t'', LessEqual (ae, r))
  | Tok_GreaterEqual ->
      let t' = match_token t Tok_GreaterEqual in
      let t'', r = parse_R t' in
      (t'', GreaterEqual (ae, r))
  | _ -> (t, ae)

and parse_AE (toks : token list) : token list * expr =
  let t, m = parse_M toks in
  match lookahead t with
  | Tok_Add ->
      let t' = match_token t Tok_Add in
      let t'', ae = parse_AE t' in
      (t'', Add (m, ae))
  | Tok_Sub ->
      let t' = match_token t Tok_Sub in
      let t'', ae = parse_AE t' in
      (t'', Sub (m, ae))
  | _ -> (t, m)

and parse_M (toks : token list) : token list * expr =
  let t, p = parse_P toks in
  match lookahead t with
  | Tok_Mult ->
      let t' = match_token t Tok_Mult in
      let t'', m = parse_M t' in
      (t'', Mult (p, m))
  | Tok_Div ->
      let t' = match_token t Tok_Div in
      let t'', m = parse_M t' in
      (t'', Div (p, m))
  | _ -> (t, p)

and parse_P (toks : token list) : token list * expr =
  let t, u = parse_U toks in
  match lookahead t with
  | Tok_Pow ->
      let t' = match_token t Tok_Pow in
      let t'', p = parse_P t' in
      (t'', Pow (u, p))
  | _ -> (t, u)

and parse_U (toks : token list) : token list * expr =
  match lookahead toks with
  | Tok_Not ->
      let t = match_token toks Tok_Not in
      let t', u = parse_U t in
      (t', Not u)
  | _ -> parse_PE toks

and parse_PE (toks : token list) : token list * expr =
  match lookahead toks with
  | Tok_Int i ->
      let t = match_token toks (Tok_Int i) in
      (t, Int i)
  | Tok_Bool b ->
      let t = match_token toks (Tok_Bool b) in
      (t, Bool b)
  | Tok_ID i ->
      let t = match_token toks (Tok_ID i) in
      (t, ID i)
  | Tok_LParen ->
      let t = match_token toks Tok_LParen in
      let t', e = parse_E t in
      let t'' = match_token t' Tok_RParen in
      (t'', e)
  | _ -> raise (InvalidInputException "problem")

let rec parse_stmt toks : stmt_result = parse_S toks

and parse_S (toks : token list) : token list * stmt =
  let look = lookahead toks in
  if look = EOF || look = Tok_RBrace then (toks, NoOp)
  else
    let t, v = parse_V toks in
    let look = lookahead t in
    if look = EOF || look = Tok_RBrace then (t, Seq (v, NoOp))
    else
      let t', v' = parse_S t in
      (t', Seq (v, v'))

and parse_V (toks : token list) : token list * stmt =
  match lookahead toks with
  | Tok_RBrace -> (toks, NoOp)
  | EOF -> (toks, NoOp)
  | Tok_Int_Type ->
      let t = match_token toks Tok_Int_Type in
      let id = lookahead t in
      let t' = match_token (match_token t id) Tok_Semi in
      let (Tok_ID str) = id in
      (t', Declare (Int_Type, str))
  | Tok_Bool_Type ->
      let t = match_token toks Tok_Bool_Type in
      let id = lookahead t in
      let t' = match_token (match_token t id) Tok_Semi in
      let (Tok_ID str) = id in
      (t', Declare (Bool_Type, str))
  | Tok_ID str ->
      let t = match_token toks (Tok_ID str) in
      let t' = match_token t Tok_Assign in
      let t'', e = parse_E t' in
      (match_token t'' Tok_Semi, Assign (str, e))
  | Tok_Print ->
      let t = match_token (match_token toks Tok_Print) Tok_LParen in
      let t', e = parse_E t in
      let t'' = match_token (match_token t' Tok_RParen) Tok_Semi in
      (t'', Print e)
  | Tok_If ->
      let t = match_token (match_token toks Tok_If) Tok_LParen in
      let t', e = parse_E t in
      let t'' = match_token (match_token t' Tok_RParen) Tok_LBrace in
      let t''', s = parse_S t'' in
      let t'''' = match_token t''' Tok_RBrace in
      if lookahead t'''' = Tok_Else then
        let t''''' = match_token (match_token t'''' Tok_Else) Tok_LBrace in
        let t'''''', s2 = parse_S t''''' in
        (match_token t'''''' Tok_RBrace, If (e, s, s2))
      else (t'''', If (e, s, NoOp))
  | Tok_For ->
      let t = match_token (match_token toks Tok_For) Tok_LParen in
      let (Tok_ID str) = lookahead t in
      let t' = match_token (match_token t (Tok_ID str)) Tok_From in
      let t'', e1 = parse_E t' in
      let t''', e2 = parse_E (match_token t'' Tok_To) in
      let t'''' = match_token (match_token t''' Tok_RParen) Tok_LBrace in
      let t''''', s = parse_S t'''' in
      (match_token t''''' Tok_RBrace, For (str, e1, e2, s))
  | Tok_While ->
      let t = match_token (match_token toks Tok_While) Tok_LParen in
      let t', e = parse_E t in
      let t'' = match_token (match_token t' Tok_RParen) Tok_LBrace in
      let t''', s = parse_S t'' in
      (match_token t''' Tok_RBrace, While (e, s))
  | _ -> raise (InvalidInputException "problem")

let parse_main toks : stmt =
  let t =
    match_token
      (match_token
         (match_token
            (match_token (match_token toks Tok_Int_Type) Tok_Main)
            Tok_LParen)
         Tok_RParen)
      Tok_LBrace
  in
  let t', s = parse_stmt t in
  let t'' = match_token (match_token t' Tok_RBrace) EOF in
  s

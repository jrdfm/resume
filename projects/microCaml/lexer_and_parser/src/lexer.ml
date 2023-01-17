open TokenTypes
open String

(* Part 1: Lexer - IMPLEMENT YOUR CODE BELOW *)

let re_intt = Str.regexp "[0-9]+"
let re_intn = Str.regexp "(\\(-[0-9]+\\))"
let re_str = Str.regexp "\"[^\"]*\""
let re_id = Str.regexp "[a-zA-Z][a-zA-Z0-9]*"
let re_rprn = Str.regexp ")"
let re_lprn = Str.regexp "("
let re_eq = Str.regexp "="
let re_neq = Str.regexp "<>"
let re_grtr = Str.regexp ">"
let re_ls = Str.regexp "<"
let re_grtreq = Str.regexp ">="
let re_lseq = Str.regexp "<="
let re_or = Str.regexp "||"
let re_and = Str.regexp "&&"
let re_add = Str.regexp "+"
let re_sub = Str.regexp "-"
let re_mult = Str.regexp "*"
let re_div = Str.regexp "/"
let re_concat = Str.regexp "\\^"
let re_arrow = Str.regexp "->"
let re_dblsmi = Str.regexp ";;"
let re_space = Str.regexp "[ \n\r\t]+"
let strip_string s = Str.global_replace (Str.regexp "[\r\n\t ]") "" s

let tokenize input =
  let rec tok pos s =
    if pos >= String.length s
    then []
    else if Str.string_match re_intn s pos
    then (
      let token = Str.matched_string s in
      let t = sub token 1 (length token - 2) in
      Tok_Int (int_of_string t) :: tok (pos + String.length token) s)
    else if Str.string_match re_rprn s pos
    then Tok_RParen :: tok (pos + 1) s
    else if Str.string_match re_lprn s pos
    then Tok_LParen :: tok (pos + 1) s
    else if Str.string_match re_id s pos
    then (
      let token = Str.matched_string s in
      match token with
      | "true" | "false" -> Tok_Bool (bool_of_string token) :: tok (pos + length token) s
      | "fun" -> Tok_Fun :: tok (pos + length token) s
      | "rec" -> Tok_Rec :: tok (pos + length token) s
      | "def" -> Tok_Def :: tok (pos + length token) s
      | "let" -> Tok_Let :: tok (pos + length token) s
      | "not" -> Tok_Not :: tok (pos + length token) s
      | "then" -> Tok_Then :: tok (pos + length token) s
      | "else" -> Tok_Else :: tok (pos + length token) s
      | "if" -> Tok_If :: tok (pos + length token) s
      | "in" -> Tok_In :: tok (pos + length token) s
      | _ -> Tok_ID token :: tok (pos + length token) s)
    else if Str.string_match re_intt s pos
    then (
      let token = Str.matched_string s in
      Tok_Int (int_of_string token) :: tok (pos + String.length token) s)
    else if Str.string_match re_space s pos
    then (
      let token = Str.matched_string s in
      tok (pos + String.length token) s)
    else if Str.string_match re_arrow s pos
    then Tok_Arrow :: tok (pos + 2) s
    else if Str.string_match re_add s pos
    then Tok_Add :: tok (pos + 1) s
    else if Str.string_match re_eq s pos
    then Tok_Equal :: tok (pos + 1) s
    else if Str.string_match re_neq s pos
    then Tok_NotEqual :: tok (pos + 2) s
    else if Str.string_match re_grtreq s pos
    then Tok_GreaterEqual :: tok (pos + 2) s
    else if Str.string_match re_lseq s pos
    then Tok_LessEqual :: tok (pos + 2) s
    else if Str.string_match re_grtr s pos
    then Tok_Greater :: tok (pos + 1) s
    else if Str.string_match re_ls s pos
    then Tok_Less :: tok (pos + 1) s
    else if Str.string_match re_or s pos
    then Tok_Or :: tok (pos + 2) s
    else if Str.string_match re_and s pos
    then Tok_And :: tok (pos + 3) s
    else if Str.string_match re_sub s pos
    then Tok_Sub :: tok (pos + 1) s
    else if Str.string_match re_mult s pos
    then Tok_Mult :: tok (pos + 1) s
    else if Str.string_match re_div s pos
    then Tok_Div :: tok (pos + 1) s
    else if Str.string_match re_concat s pos
    then Tok_Concat :: tok (pos + 1) s
    else if Str.string_match re_dblsmi s pos
    then Tok_DoubleSemi :: tok (pos + 2) s
    else if Str.string_match re_str s pos
    then (
      let token = Str.matched_string s in
      let t = Str.global_replace (Str.regexp "\"") "" token in
      Tok_String t :: tok (pos + length token) s)
    else raise (InvalidInputException "tokenize")
  in
  tok 0 input
;;

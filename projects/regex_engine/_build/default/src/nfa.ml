open List
open Sets

(*********)
(* Types *)
(*********)

type ('q, 's) transition = 'q * 's option * 'q

type ('q, 's) nfa_t = {
  sigma: 's list;
  qs: 'q list;
  q0: 'q;
  fs: 'q list;
  delta: ('q, 's) transition list;
}

(***********)
(* Utility *)
(***********)

(* explode converts a string to a character list *)
let explode (s: string) : char list =
  let rec exp i l =
    if i < 0 then l else exp (i - 1) (s.[i] :: l)
  in
  exp (String.length s - 1) []
let rec fold f a xs = match xs with
| [] -> a
| x :: xt -> fold f (f a x) xt
let contains_elem lst e = 
  fold (fun b x -> b || x=e ) false lst ;;
let rec contains l1 l2 =  
  match l1 with 
  | [] -> false
  | h::t -> if (List.mem h l2) then true else contains t l2 
;;
let rev xs = fold (fun a x -> x :: a) [] xs
let uniq lst = 
  rev (fold (fun a x-> if (contains_elem a x) then a else (x::a)) [] lst);;
let get_s e= match e with
|(_,None,_) -> None
|(n ,Some c, m) -> Some c 

let get_qo e = match e with
|(n,_, _) -> n
let get_q e = match e with
|(_,_, m) -> m

(****************)
(* Part 1: NFAs *)
(****************)
let mv nfa q s = 
  let rec aux lst = match lst with
  |[]-> []
  |h::t -> (if ((get_qo h) = q) && ((get_s h)= s) then [(get_q h)] else []) @ aux t in 
aux nfa.delta
let move (nfa: ('q,'s) nfa_t) (qs: 'q list) (s: 's option) : 'q list =
  uniq (fold (fun a x->(mv nfa x (s))@a) [] qs);;
 let rec e_closure (nfa : ('q, 's) nfa_t) (qs : 'q list) : 'q list = 
  let r' = qs in 
  let r = [] in 
  let rec aux r r' = 
    if eq r r' then r'
    else let r = r' in
         let r' = union r (move nfa r None) in
         aux r r'
  in
  aux r r'

let rec acc nfa l ls = 
  match ls with 
  | [] -> let moves = (e_closure nfa l) in 
    if (contains moves nfa.fs) then true else false
  | h::t -> acc nfa (move nfa (e_closure nfa l) (Some h)) t;;

let accept (nfa : ('q, char) nfa_t) (s : string) : bool =
  acc nfa ([nfa.q0]) (explode s);;



(*******************************)
(* Part 2: Subset Construction *)
(*******************************)

let new_states (nfa : ('q, 's) nfa_t) (qs : 'q list) : 'q list list =
  fold (fun b y ->   uniq (fold (fun a x-> (e_closure nfa (move nfa [x]  (Some y))) @ a  ) [] qs) :: b  )  [] nfa.sigma;;
  
let new_trans (nfa : ('q, 's) nfa_t) (qs : 'q list) :('q list, 's) transition list =
  fold (fun b y ->  [(qs,Some y, uniq(fold (fun a x-> (e_closure nfa (move nfa [x]  (Some y))) @a) [] qs) )] @ b )  [] nfa.sigma;;
  
let new_finals (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
  if  intersection nfa.fs qs <> [] then [qs] else [];;

let rec nfa_to_dfa_step (nfa : ('q, 's) nfa_t) (dfa : ('q list, 's) nfa_t)
  (work : 'q list list) : ('q list, 's) nfa_t =
match work with
| [] -> dfa
| h :: t -> let n = new_states nfa h in
            let f' = new_finals nfa h in
            let tr = new_trans nfa h in
            let del = union dfa.delta tr in
            let dfa ={sigma = dfa.sigma; qs = uniq(union dfa.qs [h]); q0 = dfa.q0 ; fs = union dfa.fs  f'; delta = del } in
            let t = uniq (union (minus (uniq n) dfa.qs) t) in
             nfa_to_dfa_step nfa dfa t;;
 (*let t = union(uniq (minus (n) dfa.qs)) t in*)
             
            
let nfa_to_dfa (nfa : ('q, 's) nfa_t) : ('q list, 's) nfa_t =
          let ro = e_closure nfa [ nfa.q0 ] in
          let new_st = new_states nfa ro in
          let r = union new_st [ ro ] in
          let f = new_finals nfa ro in
          let dfa = { sigma = nfa.sigma; qs = []; q0 = ro; fs = f; delta = [] } in
          nfa_to_dfa_step nfa dfa r;;
(* 
let accept (nfa : ('q, char) nfa_t) (s : string) : bool =
    let dfa = nfa_to_dfa nfa in
    let ls = explode s in
    let lst = fold (fun a x -> move dfa a (Some x)) [ dfa.q0 ] ls in
    if lst = [] then false else subset lst dfa.fs 
     *)

let get_s e= match e with
|(_,None,_) -> None
|(n ,Some c, m) -> Some c 

let get_qo e = match e with
|(n,_, _) -> n
let get_q e = match e with
|(_,_, m) -> m
let rec fold f a xs = match xs with
| [] -> a
| x :: xt -> fold f (f a x) xt


let mv nfa q s = 
  let rec aux lst = match lst with
  |[]-> []
  |h::t -> if ((get_qo h) = q) && ((get_s h)= s) then [(get_q h)] else aux t in 
aux nfa.delta


let move (nfa: ('q,'s) nfa_t) (qs: 'q list) (s: 's option) : 'q list =
  fold (fun a x->(mv nfa x (s))@a) [] qs;;

(* nfa,qs*)
let e_closure (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list = match qs with 
| [] -> []
| h::t -> (h::(mv nfa h None))@ (e_closure nfa t)

let accept (nfa: ('q,char) nfa_t) (s: string) : bool =
  let ls = explode s in 
  if (fold (fun a x -> (move nfa a (Some x)) ) [nfa.q0] ls) = nfa.fs then true else false  


  let rec e_c nfa q = match q with 
  | [] -> []
  | h::t -> (h::e_c nfa(mv nfa h None))@ (e_c nfa t)





(* 
fold (fun a x->(mv nfa_ex x s)@a) [] [0;1]
let mv nfa qs s = 
let m nfa q s = 
  let rec aux lst = match lst with
|[]-> []
| h::t -> if ((get_qo h) = q) && ((get_s h)= s) then [(get_q h)] else aux t in 
aux nfa.delta in 

fold (fun a x->(m nfa x (s))@a) [];; 

let rec e_c nfa q = match q with 
| [] -> []
| h::t -> (h::(mv nfa h None))@ (e_c nfa t)
    
*)
let nfa_e3 = {
  sigma = ['a';'b'];
  qs = ["s1";"s2";"s3";"s4"];
  q0 = "s1";
  fs = ["s4"];
  delta = [("s1",None, "s3"); ("s1", None, "s2");("s2",Some 'a',"s2");
          ("s2",None,"s4");("s3",Some 'b',"s3");("s3",None,"s4")]
};;

let nfa_e4 = {
  sigma = ['a';'b'];
  qs = [1;2;3];
  q0 = 1;
  fs = [3];
  delta = [(1,Some 'a', 2); (1, Some 'a', 1);(1,Some 'b',1);(2;Some 'b',3)]
};;

let m' = {
  sigma = nfa_e1.sigma;
  qs = [qo'];
  q0 = qo';
  fs = f';
  delta = []

};;



let rec nfa_to_dfa_step (nfa : ('q, 's) nfa_t) (dfa : ('q list, 's) nfa_t)
    (work : 'q list list) : ('q list, 's) nfa_t =
  match work with
  | [] -> dfa
  | h :: t -> let n = nxt nfa h nfa.sigma in
              let s = e_closure nfa n in
              let f' = new_finals nfa s in
              let tr = fold (fun a x -> if (e_closure nfa (move nfa h (Some x))) = s then [(h,Some x,s)] else a) [] nfa.sigma in
              let del = union dfa.delta tr in
              let dfa ={sigma = nfa.sigma; qs = dfa.qs; q0 = dfa.q0 ; fs = union dfa.fs  f'; delta = del } in
              nfa_to_dfa_step nfa dfa t;;
      

let nfa_to_dfa (nfa : ('q, 's) nfa_t) : ('q list, 's) nfa_t =
  let ro = e_closure nfa [ nfa.q0 ] in
  let new_st = new_states nfa ro in
  let r = union new_st [ ro ] in
  let f = new_finals nfa ro in

  let dfa = { sigma = nfa.sigma; qs = r; q0 = ro; fs = f; delta = [] } in
  nfa_to_dfa_step nfa dfa r;;


(* 
let rec nfa_to_dfa_step (nfa : ('q, 's) nfa_t) (dfa : ('q list, 's) nfa_t)
    (work : 'q list list) : ('q list, 's) nfa_t =
  match work with
  | [] -> dfa
  | h :: t -> let n = nxt nfa h nfa.sigma in
              let s = e_closure nfa n in
              let f' = new_finals nfa s in

              let tr = fold (fun a x -> if (move nfa h x) = s then [(h,Some x,s)] else a) [] nfa.sigma in 
              let tr = fold (fun a x -> if (e_closure nfa (move nfa h (Some x))) = s then [(h,Some x,s)] else a) [] nfa.sigma in
              let del = union dfa.delta ( tr nfa h (nxt nfa h nfa.sigma) (new_trans nfa h) ) in
              let dfa ={sigma = nfa.sigma; qs = dfa.qs; q0 = dfa.q0 ; fs = union dfa.fs  f'; delta = del } in
              nfa_to_dfa_step nfa dfa t;; *)

  
  nfa_to_dfa nfa_ex;;
  - : (int list, char) nfa_t =
  {sigma = ['a']; qs = [[1; 2]; [0]]; q0 = [0]; fs = [[1; 2]];
   delta = [([0], Some 'a', [1])]}




(* ─( 20:16:01 )─< command 15 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # tr nfa_e1 ["p1";"p3"] ["p2"] (new_trans nfa_e1 ["p1";"p3"]);;
   - : (string list, char) transition = (["p1"; "p3"], Some 'a', ["p2"])
   ─( 20:16:01 )─< command 16 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # tr nfa_e1 ["p1";"p2"] (nxt nfa_e1  ["p1";"p2"] nfa_e1.sigma) (new_trans nfa_e1 ["p1";"p3"]);;
   Exception: Failure "sth".
   ─( 20:16:55 )─< command 17 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # tr nfa_e1 ["p1";"p3"] (nxt nfa_e1  ["p1";"p3"] nfa_e1.sigma) (new_trans nfa_e1 ["p1";"p3"]);;
   - : (string list, char) transition = (["p1"; "p3"], Some 'a', ["p2"])
   ─( 20:17:29 )─< command 18 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # let ro = e_closure nfa_e1 [nfa_e1.q0];;
   val ro : string list = ["p1"; "p3"]
   ─( 20:17:48 )─< command 19 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # ro;;
   - : string list = ["p1"; "p3"]
   ─( 20:18:33 )─< command 20 >────────────────────────────────────────────────────────────────────────────────────────{ counter: 0 }─
   utop # tr nfa_e1 ro (nxt nfa_e1 ro nfa_e1.sigma) (new_trans nfa_e1 ro);;
   - : (string list, char) transition = (["p1"; "p3"], Some 'a', ["p2"])
   ─( 20:18:38 )─< command 21 >───────────────────────────────────────────
*)
(*  Trash *)


(* let rec nfa_to_dfa_step (nfa : ('q, 's) nfa_t) (dfa : ('q list, 's) nfa_t)
    (work : 'q list list) : ('q list, 's) nfa_t =
  match work with
  | [] -> dfa
  | h :: t -> let n = nxt nfa h nfa.sigma in
              let s = e_closure nfa n in
              if elem s work = false then
                let r = union work [ s ] in
                let f' = new_finals nfa s in
                let del = union dfa.delta (tr nfa h (nxt nfa h nfa.sigma) (new_trans nfa h))  in
                let dfa ={sigma = nfa.sigma; qs = r; q0 = [ nfa.q0 ]; fs = union [ nfa.fs ] f'; delta = del } in
                dfa
              else nfa_to_dfa_step nfa dfa t;;
      

let nfa_to_dfa (nfa : ('q, 's) nfa_t) : ('q list, 's) nfa_t =
  let ro = e_closure nfa [ nfa.q0 ] in
  let new_st = new_states nfa ro in
  let r = union new_st [ ro ] in
  let f = new_finals nfa ro in

  let dfa = { sigma = nfa.sigma; qs = r; q0 = ro; fs = f; delta = [] } in

  nfa_to_dfa_step nfa dfa r;;

  *)



(*
       let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
         (fold (fun a x -> if (get_qo x) = a then nw nfa (get_qo x) else a) x nfa.delta) ;;




       fold (fun a x -> ((nw nfa x))::a) [qs] qs;;


       fold (fun a x -> (nw nfa x  ):: a ) [] qs;;


       fold (fun a x -> if (get_qo x) = q then nw nfa q else []) q nfa.delta

       - : int list list = [[1; 2]]

       fold (fun b x -> (e_closure nfa_ex (move nfa_ex [x] (Some 'a')))::b) [] [0];;

       let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
    fold (fun a x -> fold (fun a x -> (e_closure nfa x)::a ) [] (nw nfa x)) [] qs;;

    let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
     rm_e (fold (fun a x -> fold (fun a x -> (e_closure nfa x)::a ) [] (nw nfa x)) [] qs);;

     let rec e_closure (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list = match qs with
   | [] -> []
   | h::t -> uniq ((h::(mv nfa h None))@ (e_closure nfa t))


   let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
     fold (fun a x -> ((nw nfa x))::a) [] qs;;

     let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
       fold (fun b x -> (e_closure nfa (nw nfa x))::b) [] qs;;


   let new_states (nfa: ('q,'s) nfa_t) (qs: 'q list) : 'q list list =
     fold (fun b x -> (e_closure nfa (move nfa [x] (Some 'a')))::b) [] qs;;

     let nw nfa q=
     let rec aux lst = match lst with
     |[]-> []
     |h::t -> (if ((get_qo h) = q) then [(get_q h)] else []) @ aux t in
   aux nfa.delta


   let mv nfa q s =
     let rec aux lst = match lst with
     |[]-> []
     |h::t -> if ((get_qo h) = q) && ((get_s h)= s) then [(get_q h)] else aux t in
   aux nfa.delta
*)

(*
   let accept (nfa: ('q,char) nfa_t) (s: string) : bool =
   let ls = explode s in
   if (fold (fun a x -> (move nfa a (Some x)) ) [nfa.q0] ls) = nfa.fs then true else false *)


   (* More Trash  *)


   (*  
   let rec nfa_to_dfa_step (nfa : ('q, 's) nfa_t) (dfa : ('q list, 's) nfa_t)
    (work : 'q list list) : ('q list, 's) nfa_t =
  match work with
  | [] -> dfa
  | h :: t -> let n = nxt nfa h nfa.sigma in
              let s = e_closure nfa n in
              let f' = new_finals nfa s in
              let del = union dfa.delta ( tr nfa h (nxt nfa h nfa.sigma) (new_trans nfa h) ) in
              let dfa ={sigma = nfa.sigma; qs = dfa.qs; q0 = dfa.q0 ; fs = union dfa.fs  f'; delta = del } in
              nfa_to_dfa_step nfa dfa t;;
      

let nfa_to_dfa (nfa : ('q, 's) nfa_t) : ('q list, 's) nfa_t =
  let ro = e_closure nfa [ nfa.q0 ] in
  let new_st = new_states nfa ro in
  let r = union new_st [ ro ] in
  let f = new_finals nfa ro in

  let dfa = { sigma = nfa.sigma; qs = r; q0 = ro; fs = f; delta = [] } in

  nfa_to_dfa_step nfa dfa r;;
   
   
   *)
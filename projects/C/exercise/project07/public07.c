#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 7 (public07.c)
 *
 * Tests adding elements back to a priority queue from which all elements
 * were removed, then clears it and checks that there are no memory leaks or
 * errors in the heap.  This test is basically the same as the ninth Project
 * #6 public test, with memory checking added.
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the project assignment for explanation).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *elements[]= {"Rocky Racoon", "Wally Walrus", "Aaron Aardvark",
                     "Ginny Giraffe", "Manny Manatee", "Donny Donkey",
                     "Courtney Koala", "Bunny Rabbit"};
  int i, priorities[]= {90, 40, 60, 20, 30, 50, 80, 70};
  char *result;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  assert(size(prio_q) == 8);

  for (i= 1; i <= 8; i++) {
    result= dequeue(&prio_q);
    assert(result != NULL);
    free(result);
 }

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  assert(enqueue(&prio_q, "Horace Horse", 97) == 1);
  assert(enqueue(&prio_q, "Hammy Hamster", 53) == 1);
  assert(enqueue(&prio_q, "Otto Otter", 29) == 1);
  assert(enqueue(&prio_q, "Dolly Dolphin", 74) == 1);

  assert(size(prio_q) == 4);

  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}

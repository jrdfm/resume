#include <stdio.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 8 (public08.c)
 *
 * Tests calling get_all_elements() on a priority queue that has some
 * elements, then clears it and checks that there are no memory leaks or
 * errors in the heap.  This test is basically the same as the eleventh
 * Project #6 public test, with memory checking added.
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
  int priorities[]= {90, 40, 60, 20, 30, 50, 80, 70};
  char *expected_elements[]= {"Rocky Racoon", "Courtney Koala", "Bunny Rabbit",
                              "Aaron Aardvark", "Donny Donkey", "Wally Walrus",
                              "Manny Manatee", "Ginny Giraffe", NULL};
  char **result;
  int i;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  /* the array expected_elements has the element that were added but they
     are in order of priority, and the last element is NULL, which
     compare_name_lists() requires */
  result= get_all_elements(prio_q);
  assert(compare_name_lists(result, expected_elements) == 1);

  /* now clean up after ourselves */
  free_name_list(result);
  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}

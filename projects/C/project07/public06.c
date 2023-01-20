#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 6 (public06.c)
 *
 * Tests adding a few elements to a priority queue and verifies that they
 * are removed in order of priority, then clears it and checks that there
 * are no memory leaks or errors in the heap.  This test is basically the
 * same as the fourth Project #6 public test, with memory checking added.
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
                     "Ginny Giraffe", "Manny Manatee", "Donny Donkey"};
  int priorities[]= {90, 40, 60, 20, 30, 50};
  char *element;
  int i;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  element= dequeue(&prio_q);
  assert(strcmp(element, "Rocky Racoon") == 0);
  free(element);  /* now clean up after ourselves */
  element= dequeue(&prio_q);
  assert(strcmp(element, "Aaron Aardvark") == 0);
  free(element);  /* now clean up after ourselves */
  element= dequeue(&prio_q);
  assert(strcmp(element, "Donny Donkey") == 0);
  free(element);  /* now clean up after ourselves */
  element= dequeue(&prio_q);
  assert(strcmp(element, "Wally Walrus") == 0);
  free(element);  /* now clean up after ourselves */
  element= dequeue(&prio_q);
  assert(strcmp(element, "Manny Manatee") == 0);
  free(element);  /* now clean up after ourselves */

  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}

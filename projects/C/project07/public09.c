#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 9 (public09.c)
 *
 * Tests calling enqueue() with different orderings of the elements'
 * priorities, then clears it and checks that there are no memory leaks or
 * errors in the heap.  This test is basically the same as the fourth
 * Project #6 secret test, with memory checking added.
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the project assignment for explanation).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

static void check_all_elements(Prio_que *prio_q, char *expected_elements[]);

/* helper function to check all elements in a queue without calling
 * get_all_elements(), just so someone can pass this test even if their
 * get_all_elements() has a bug.
 */
static void check_all_elements(Prio_que *prio_q, char *expected_elements[]) {
  char *element;
  int i= 0;

  while (!is_empty(*prio_q)) {
    element= dequeue(prio_q);
    assert(strcmp(element, expected_elements[i++]) == 0);
    free(element);  /* now clean up after ourselves */
  }
}

int main(void) {
  Prio_que prio_q;
  char *elements1[]= {"Aaron Aardvark", "Ginny Giraffe", "Rocky Racoon",
                      "Wally Walrus", "Donny Donkey", "Manny Manatee"},
       *elements2[]= {"Aaron Aardvark", "Ginny Giraffe", "Wally Walrus",
                      "Donny Donkey", "Manny Manatee", "Rocky Racoon"},
       *expected_elements[]= {"Rocky Racoon", "Aaron Aardvark", "Donny Donkey",
                              "Wally Walrus", "Manny Manatee", "Ginny Giraffe"};
  int priorities1[]= {60, 20, 90, 40, 50, 30},
      priorities2[]= {60, 20, 40, 50, 30, 90};
  int i;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements1); i++)
    enqueue(&prio_q, elements1[i], priorities1[i]);

  check_all_elements(&prio_q, expected_elements);  /* will empty the queue */

  /* add elements with different priorities this time */
  for (i= 0; i < ARRSIZE(elements2); i++)
    enqueue(&prio_q, elements2[i], priorities2[i]);

  check_all_elements(&prio_q, expected_elements);

  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}

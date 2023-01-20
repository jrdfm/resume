#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 1 (public01.c)
 *
 * Tests that clear() removes all of a priority queue's elements.
 *
 * This test will always unavoidably have memory leaks if you run it under
 * valgrind, so fix any other problems that valgrind identifies, but ignore
 * any memory leaks (or just add the --leak-check=no argument when running
 * this test with valgrind).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *elements[]= {"Rocky Racoon", "Wally Walrus", "Aaron Aardvark",
                     "Ginny Giraffe", "Manny Manatee", "Donny Donkey", 
                     "Courtney Koala"};
  int i, priorities[]= {90, 40, 60, 20, 30, 50, 80};

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  assert(size(prio_q) == 7);

  clear(&prio_q);
  init(&prio_q);
  
  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Yet another test worked right!\n");

  return 0;
}

#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 5 (public05.c)
 *
 * Tests a priority queue's size after removing some elements.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *elements[]= {"Rocky Racoon", "Wally Walrus", "Aaron Aardvark",
                     "Ginny Giraffe", "Manny Manatee", "Donny Donkey",
                     "Courtney Koala"};
  int priorities[]= {90, 40, 60, 20, 30, 50, 80};
  int i;

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  for (i= 1; i <= 4; i++)
    dequeue(&prio_q);

  assert(size(prio_q) == 3);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

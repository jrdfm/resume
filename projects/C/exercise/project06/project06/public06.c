#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 6 (public06.c)
 *
 * Tests that elements with duplicate priority cannot be added to a priority
 * queue.
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

  assert(size(prio_q) == 7);

  assert(enqueue(&prio_q, "Ellie Elephant", 60) == 0);  /* this should fail */
  assert(size(prio_q) == 7);  /* the size should not have changed */

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

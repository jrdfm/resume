#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 4 (public04.c)
 *
 * Tests adding a few elements to a priority queue and verifies that they
 * are removed in order of priority.
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

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  element= dequeue(&prio_q);
  assert(strcmp(element, "Rocky Racoon") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Aaron Aardvark") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Donny Donkey") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Wally Walrus") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Manny Manatee") == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

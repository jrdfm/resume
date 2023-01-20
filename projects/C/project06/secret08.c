#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 8 (secret08.c)
 *
 * Tests that peek() returns a copy of the data in the queue and not just an
 * alias to it.
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

  element= peek(prio_q);
  element[0]= 'Z';  /* this should not change the data in the queue itself */

  element= peek(prio_q);
  assert(strcmp(element, "Rocky Racoon") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Rocky Racoon") == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 13 (secret13.c)
 *
 * Tests calling enqueue() passing NULL into new_element.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *element;

  init(&prio_q);

  enqueue(&prio_q, "Rocky Racoon", 90);
  enqueue(&prio_q, "Aaron Aardvark", 60);
  enqueue(&prio_q, NULL, 40);  /* should have no effect */
  enqueue(&prio_q, "Ginny Giraffe", 20);

  element= dequeue(&prio_q);
  assert(strcmp(element, "Rocky Racoon") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Aaron Aardvark") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Ginny Giraffe") == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

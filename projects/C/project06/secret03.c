#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 3 (secret03.c)
 *
 * Tests that enqueue() is allocating new memory for elements being stored,
 * and is not just performing pointer aliasing.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char name[20]= "Hammy Hamster", *element;

  init(&prio_q);

  enqueue(&prio_q, "Myrtle Turtle", 35);
  enqueue(&prio_q, name, 25);
  enqueue(&prio_q, "Ryan Lion", 15);

  /* now change name, which should not change anything being stored in the
     queue unless the queue elements were incorrectly just aliased to the
     arguments passed into enqueue() */
  strcpy(name, "Horace Horse");

  element= dequeue(&prio_q);
  assert(strcmp(element, "Myrtle Turtle") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Hammy Hamster") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "Ryan Lion") == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

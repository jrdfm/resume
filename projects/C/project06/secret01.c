#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 1 (secret01.c)
 *
 * Tests calling init() to reset a priority queue, and adds new elements
 * afterwards.
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
  int i;

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  assert(size(prio_q) == 8);

  init(&prio_q);  /* remove everything */

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  assert(enqueue(&prio_q, "Horace Horse", 97) == 1);
  assert(enqueue(&prio_q, "Hammy Hamster", 53) == 1);
  assert(enqueue(&prio_q, "Otto Otter", 29) == 1);
  assert(enqueue(&prio_q, "Dolly Dolphin", 74) == 1);

  assert(size(prio_q) == 4);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

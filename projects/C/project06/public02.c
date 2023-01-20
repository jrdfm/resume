#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 2 (public02.c)
 *
 * Tests adding a few elements to a priority queue and checks its size.
 *
 * To reduce code (here and in other tests) and make it easier to vary
 * tests, the elements and priorities to be added are stored in two arrays,
 * with element i of the priority array being the priority for element i of
 * the element array.  You can do the smae in your own tests but of course
 * it only works if the two arrays are the same size.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *elements[]= {"Rocky Racoon", "Wally Walrus", "Aaron Aardvark",
                     "Ginny Giraffe", "Manny Manatee", "Donny Donkey"};
  int priorities[]= {90, 40, 60, 20, 30, 50};
  int i;

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  assert(size(prio_q) == 6);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

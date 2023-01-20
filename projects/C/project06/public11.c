#include <stdio.h>
#include <assert.h>
#include "prio-q.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 11 (public11.c)
 *
 * Tests calling get_all_elements() on a priority queue that has some
 * elements.
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
  char *expected_elements[]= {"Rocky Racoon", "Courtney Koala", "Bunny Rabbit",
                              "Aaron Aardvark", "Donny Donkey", "Wally Walrus",
                              "Manny Manatee", "Ginny Giraffe", NULL};
  int i;

  init(&prio_q);

  for (i= 0; i < (int) (sizeof(elements) / sizeof(elements[0])); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  /* the array expected_elements has the element that were added but they
     are in order of priority, and the last element is NULL, which
     compare_name_lists() requires */
  assert(compare_name_lists(get_all_elements(prio_q), expected_elements) == 1);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

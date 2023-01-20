#include <stdio.h>
#include <assert.h>
#include "prio-q.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 7 (secret07.c)
 *
 * Creates multiple priority queues, to ensure that their data doesn't
 * conflict.  (If you aren't passing this test you may be using global
 * variables, which you weren't supposed to do.)
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q1, prio_q2;
  char *elements1[]= {"Rocky Racoon", "Wally Walrus", "Aaron Aardvark",
                      "Ginny Giraffe", "Manny Manatee", "Donny Donkey",
                      "Courtney Koala"},
       *elements2[]= {"Bruce Goose", "Ellie Elephant", "Perry Parrot",
                      "Sally Salamander", "Leonard Leopard"};
  int priorities1[]= {90, 40, 60, 20, 30, 50, 80},
      priorities2[]= {75, 50, 10, 25, 30}, i;
  char *expected_elements1[]= {"Rocky Racoon", "Courtney Koala",
                               "Aaron Aardvark", "Donny Donkey",
                               "Wally Walrus", "Manny Manatee",
                               "Ginny Giraffe", NULL};
  char *expected_elements2[]= {"Bruce Goose", "Ellie Elephant",
                               "Leonard Leopard", "Sally Salamander",
                               "Perry Parrot", NULL};

  init(&prio_q1);
  init(&prio_q2);

  for (i= 0; i < (int) (sizeof(elements1) / sizeof(elements1[0])); i++)
    enqueue(&prio_q1, elements1[i], priorities1[i]);

  for (i= 0; i < (int) (sizeof(elements2) / sizeof(elements2[0])); i++)
    enqueue(&prio_q2, elements2[i], priorities2[i]);

  assert(compare_name_lists(get_all_elements(prio_q1),
                            expected_elements1) == 1);
  assert(compare_name_lists(get_all_elements(prio_q2),
                            expected_elements2) == 1);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}

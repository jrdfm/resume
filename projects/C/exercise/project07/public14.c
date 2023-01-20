#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 14 (public14.c)
 *
 * Tests the basic operation of remove_elements_between().
 *
 * This test will always unavoidably have memory leaks if you run it under
 * valgrind, so fix any other problems that valgrind identifies, but ignore
 * any memory leaks (or just add the --leak-check=no argument when running
 * this test with valgrind).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *elements[]= {"Rocky Racoon", "Ellie Elephant", "Wally Walrus",
                     "Aaron Aardvark", "Perry Parrot", "Bruce Goose",
                     "Ginny Giraffe", "Sally Salamander", "Manny Manatee",
                     "Donny Donkey", "Leonard Leopard", "Courtney Koala"};
  int i, priorities[]= {90, 55, 40, 60, 10, 75, 20, 25, 30, 50, 45, 80};
  char *expected_elements[]= {"Rocky Racoon", "Courtney Koala", "Bruce Goose",
                              "Aaron Aardvark", "Ellie Elephant",
                              "Leonard Leopard", "Wally Walrus",
                              "Manny Manatee", "Sally Salamander",
                              "Ginny Giraffe", "Perry Parrot", NULL};
  char **result;

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  remove_elements_between(&prio_q, 50, 50);

  result= get_all_elements(prio_q);
  assert(compare_name_lists(result, expected_elements) == 1);

  printf("Yet another test worked right!\n");

  return 0;
}

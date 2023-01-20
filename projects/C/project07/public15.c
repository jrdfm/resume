#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 15 (public15.c)
 *
 * Tests the basic operation of change_priority().
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
  char *elements[]= {"Rocky Racoon", "Earl Squirrel", "Sally Salamander",
                     "Shelly Sheep", "Robin Robin", "Giselle Gazelle",
                     "Neil Seal", "Placido Flamingo"};
  int priorities[]= {90, 40, 60, 20, 30, 50, 70, 80};
  char *expected_elements[]= {"Rocky Racoon", "Neil Seal", "Sally Salamander",
                              "Giselle Gazelle", "Placido Flamingo",
                              "Earl Squirrel", "Robin Robin", "Shelly Sheep"};
  int expected_priorities[]= {90, 70, 60, 50, 45, 40, 30, 20};
  int i;

  init(&prio_q);

  for (i= 0; i < ARRSIZE(elements); i++)
    enqueue(&prio_q, elements[i], priorities[i]);

  assert(change_priority(&prio_q, "Placido Flamingo", 45) == 1);

  for (i= 0; i < ARRSIZE(expected_elements); i++)
    assert(get_priority(prio_q, expected_elements[i]) ==
           expected_priorities[i]);

  printf("Yet another test worked right!\n");

  return 0;
}

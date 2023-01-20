#include <stdio.h>
#include <assert.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 7 (public07.c)
 *
 * Tests compare().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student1= {NULL, NULL}, student2= {NULL, NULL};

  init_student(&student1, "Shelly", "Sheep");
  init_student(&student2, "Goldie", "Goat");

  assert(compare(student1, student1) == 0);
  assert(compare(student1, student2) > 0);

  printf("Affirmative!\n");

  return 0;
}

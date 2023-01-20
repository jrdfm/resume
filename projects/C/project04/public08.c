#include <stdio.h>
#include <assert.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 8 (public08.c)
 *
 * Tests calling reset_student();
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  init_student(&student, "Paris", "Parrot");

  reset_student(&student);

  /* both names should be NULL now, and there should not be any errors */
  assert(student.first_name == NULL);
  assert(student.last_name == NULL);

  printf("Affirmative!\n");

  return 0;
}

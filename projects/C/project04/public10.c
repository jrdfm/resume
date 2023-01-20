#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 10 (public10.c)
 *
 * Tests calling init_student() with NULL names.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  init_student(&student, "Otto", "Otter");

  /* this should have no effect, and definitely not crash */
  init_student(&student, NULL, NULL);

  assert(strcmp(student.first_name, "Otto") == 0);
  assert(strcmp(student.last_name, "Otter") == 0);

  printf("Affirmative!\n");

  return 0;
}

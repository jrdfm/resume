#include <stdio.h>
#include <assert.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 9 (public09.c)
 *
 * Tests calling free_student();
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student *student= create_student("Paris", "Parrot");

  if (student != NULL) {
    free_student(&student);
    assert(student == NULL);
  }

  printf("Affirmative!\n");

  return 0;
}

#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 5 (public05.c)
 *
 * Tests calling change_last_name().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  init_student(&student, "Portia", "Porcupine");
  change_last_name(&student, "Porpoise");

  print_student(student);
  printf("\n");

  return 0;
}

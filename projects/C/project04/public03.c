#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 3 (public03.c)
 *
 * Tests creating several students to ensure that their data does not
 * conflict.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student1= {NULL, NULL}, student2= {NULL, NULL},
          student3= {NULL, NULL};

  init_student(&student1, "Hammy", "Hamster");
  init_student(&student2, "Gerry", "Gerbil");
  init_student(&student3, "Chippy", "Chipmunk");

  print_student(student1);
  printf("\n");

  print_student(student2);
  printf("\n");

  print_student(student3);
  printf("\n");

  return 0;
}

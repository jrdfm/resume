#include "student.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_student(Student *const student, const char first_name[],
                  const char last_name[]) {

  if (student != NULL && first_name != NULL && last_name != NULL) {
    char *first = malloc((strlen(first_name) + 1) * sizeof(char));
    char *last = malloc((strlen(last_name) + 1) * sizeof(char));
    strcpy(first, first_name);
    strcpy(last, last_name);
    student->first_name = first;
    student->last_name = last;
  }
}

void print_student(Student student) {

  if (student.last_name != NULL) {
    printf("%s%s", student.last_name, ", ");
  }

  if (student.first_name != NULL) {
    printf("%s", student.first_name);
  }
}

Student *create_student(const char first_name[], const char last_name[]) {

  if (first_name != NULL && last_name != NULL) {
    Student *new = malloc(sizeof(Student));
    char *first = malloc((strlen(first_name) + 1) * sizeof(char));
    char *last = malloc((strlen(last_name) + 1) * sizeof(char));
    strcpy(first, first_name);
    strcpy(last, last_name);

    new->first_name = first;
    new->last_name = last;
    return new;
  }

  return NULL;
}

unsigned int change_first_name(Student *const student, const char new_name[]) {

  if (student != NULL && new_name != NULL) {
    char *first = malloc((strlen(new_name) + 1) * sizeof(char));

    strcpy(first, new_name);
    free(student->first_name);
    student->first_name = first;

    return 1;
  }

  return 0;
}
unsigned int change_last_name(Student *const student, const char new_name[]) {

  if (student != NULL && new_name != NULL) {
    char *last = malloc((strlen(new_name) + 1) * sizeof(char));

    strcpy(last, new_name);
    free(student->last_name);
    student->last_name = last;

    return 1;
  }

  return 0;
}

unsigned int add_nickname(Student *const student, const char nickname[]) {

  if (student != NULL && nickname != NULL &&
      (strchr(student->first_name, '(') == NULL)) {
    char si[] = {" ("};
    char sf[] = {")"};

    char *new = malloc((strlen(student->first_name) + strlen(nickname) + 4) *
                       sizeof(char));

    strcpy(new, student->first_name);
    strcat(new, si);
    strcat(new, nickname);
    strcat(new, sf);
    free(student->first_name);
    student->first_name = new;

    return 1;
  }

  return 0;
}

int compare(Student student1, Student student2) {

  if (strcmp(student1.last_name, student2.last_name) == 0) {
    return strcmp(student1.first_name, student2.first_name);
  }

  else {
    return strcmp(student1.last_name, student2.last_name);
  }
}

unsigned int reset_student(Student *const student) {

  student->first_name = NULL;
  student->last_name = NULL;

  free(student->first_name);
  free(student->last_name);

  return 1;
}

unsigned int free_student(Student **const student) {

  if (student != NULL && *student != NULL) {
    (*student)->first_name = NULL;
    (*student)->last_name = NULL;
    free(*student);
    *student = NULL;
    return 1;
  }
  return 0;
}
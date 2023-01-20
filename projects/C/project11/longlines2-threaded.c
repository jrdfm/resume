#include "randomdelay.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 */

/* This program processes a list of zero or more files whose names are given
 * on its command line, counting the number of lines that each file contains
 * that are longer than 80 characters.  It aggregates the number of lines in
 * all the files that are longer than 80 characters and prints the total as
 * its only output.  Files are read one character at a time, by character, *
 * because we don't have any way to know the maximum length of lines.  More
 * realistically, and unlike Project #2, we do not require that the last
 * line of a file end with a newline.
 */

static void *check_lines(void *ptr);

static void *check_lines(void *ptr) {
  int line_length, count;
  char *file_name, ch;
  FILE *fp;
  int *re = malloc(sizeof(int));

  file_name = (char *)ptr;

  fp = fopen(file_name, "r"); /* open file for reading */
  if (fp == NULL) {
    return 0;
  }
  if (fp != NULL) {
    count = 0;
    line_length = 0;

    /* see the explanation in the project assignment regarding what this
       is for */

    while ((ch = fgetc(fp)) != EOF) { /* read by character */
      if (ch != '\n')
        line_length++;
      else {
        if (line_length > 80)
          count++;
        line_length = 0; /* reset for the next line */
      }
    }

    /* we have to handle it as a special case if the last line does not
       end with a newline and it is longer than 80 characters (of course
       only the last line can possibly not end with a newline) */
    if (line_length > 80)
      count++;
  }

  fclose(fp);
  *re = count;
  pthread_exit((void *)re);
}

int main(int argc, char *argv[]) {
  int all_files_total = 0;
  int i;
  int *one_file_count;
  pthread_t *tids;

  tids = malloc(sizeof(pthread_t) * argc);

  if (argc > 1) { /* if there was at least one filename provided */
    /* iterate over all filenames on the command line */
    for (i = 1; i < argc; i++) {
      pthread_create(&tids[i], NULL, &check_lines, (void *)(argv[i]));
    }
  }

  for (i = 1; i < argc; i++) {
    pthread_join(tids[i], (void **)&one_file_count);

    if (one_file_count != NULL) {
      all_files_total += *one_file_count;
    }
    free(one_file_count);
  }

  printf("%d\n", all_files_total);
  free(tids);

  return 0;
}

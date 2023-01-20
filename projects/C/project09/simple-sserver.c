#include "simple-sserver.h"
#include "safe-fork.h"
#include "simple-sserver-datastructure.h"
#include "split.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sysexits.h>
#include <unistd.h>
#include <wait.h>

static void clear(Node *const head);
static void free_wrd(char *name_list[]);
static char **remove_word(int n, char **array);

/* CMSC 216, Spring 2021, Project #9
 * simple-sserver.c
 *
 * I pledge on my honor that I have not given or received any unauthorized
 * assistance on this assignment.
 *
 * Yared Fikremariam
 * TerpConnect ID: yfikrema
 * Section: 0302 UID: 116945769
 *
 * This program contains functions,defined in simple-sserver.h. read_commads
 * reads two sets of commands from file,one for compilation & another for
 * test. clear_commands deallocates all memory in its parameter points to.
 * compile_program runs commands in compilation list and test_program runs
 * commands in the test list.
 *
 */

/* This function reads two sets of commands from files and initializes
 * & returns Commands which is a type containing two dynamically allocated
 * linked lists of the read sets of commands.
 * It takes in the name of the two files and exits if either of its parameters
 * are NULL, files doesn't exist or should any IO error occurs.
 *
 */

Commands read_commands(const char compile_cmds[], const char test_cmds[]) {

  Commands cmds;
  Node *new, *curr;
  char line[LINE_MAX];
  FILE *fc, *ft;
  fc = fopen(compile_cmds, "r");
  ft = fopen(test_cmds, "r");
  cmds.cmp_head = NULL;
  cmds.tst_head = NULL;

  if (fc == NULL || ft == NULL) { /* exit if either file is NULL */
    exit(1);
  }

  while (fgets(line, LINE_MAX, fc) != NULL) {
    new = malloc(sizeof(*new));

    if (new != NULL) {
      /* allocate appropriate amount of memory */
      new->data = malloc(strlen(line) + 1);
      strcpy(new->data, line);
      new->next = NULL;
      if (cmds.cmp_head == NULL) { /* head case */
        cmds.cmp_head = new;
        curr = new;
      } else {
        curr = curr->next = new;
      }
    }
  }

  fclose(fc); /* close the compile_commands file */

  while (fgets(line, LINE_MAX, ft) != NULL) {
    new = malloc(sizeof(*new));
    if (new != NULL) {
      /* allocate appropriate amount of memory */
      new->data = malloc(strlen(line) + 1);
      strcpy(new->data, line);
      new->next = NULL;
      if (cmds.tst_head == NULL) { /* head case */
        cmds.tst_head = new;
        curr = new;
      } else {
        curr = curr->next = new;
      }
    }
  }

  fclose(ft); /* close the test_commands file */

  return cmds;
}

/* This function frees all the dynamically allocated memory in its argument.
 * If its parameter is NULL it returns without changing anything.
 */

void clear_commands(Commands *const commands) {

  if (commands != NULL) {
    clear(commands->cmp_head);
    clear(commands->tst_head);
  }
  return;
}

/* This function runs the compilation commands it was passed in its parameter.
 * It returns SUCCESSFUL_COMPILATION if all commands were successful, exit with
 * 0, or halts & returns FAILED_COMPILATION if any of its commands had an error.
 */

int compile_program(Commands commands) {

  Node *curr;
  char **words;
  pid_t pid;
  int status, fdi, fdo;
  fdi = dup(STDIN_FILENO);  /* duplicate file descriptor for stdin */
  fdo = dup(STDOUT_FILENO); /* duplicate file descriptor for stdout */
  curr = commands.cmp_head;

  while (curr != NULL) {
    words = split(curr->data);
    dup2(fdi, STDIN_FILENO);  /* restore to input to stdin */
    dup2(fdo, STDOUT_FILENO); /* restore to output to stdout */

    pid = safe_fork(); /* create a child */

    if (pid == 0) {  /* child */
      execvp(words[0], words); /* run the command */

    } else if (pid > 0) { /* parent */

      free_wrd(words); /* free array returned by split */
      wait(&status);
      if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) == 1) {
          return FAILED_COMPILATION;
        }
      }
    }
    curr = curr->next;
  }
  return SUCCESSFUL_COMPILATION;
}

/* This function runs the test commands it was passed in its parameter.But
 * unlike compile_program it runs to completion even if there was an error with
 * one or more of its commands. It returns the number of successful commands.
 */

int test_program(Commands commands) {

  Node *curr;
  char **words;
  pid_t pid;
  int status, count, i;
  char infile[LINE_MAX], outfile[LINE_MAX];
  int fdi, fdo;
  fdi = dup(STDIN_FILENO);  /* duplicate file descriptor for stdin */
  fdo = dup(STDOUT_FILENO); /* duplicate file descriptor for stdout */

  curr = commands.tst_head;

  while (curr != NULL) {
    words = split(curr->data);
    /* Incase command contains IO redirection*/
    if (strchr(curr->data, '<') != NULL || (strchr(curr->data, '>') != NULL)) {
      i = 0;
      while (words[i] != NULL) {
        if (strcmp(words[i], "<") == 0) {
          strcpy(infile, words[i + 1]);  /* get input filename */
          words = remove_word(i, words); /* remove < and filename */
          fdi = open(infile, O_RDONLY);
        }
        if ((words[i] != NULL) && strcmp(words[i], ">") == 0) {

          strcpy(outfile, words[i + 1]); /* get output filename */
          words = remove_word(i, words); /* remove > and filename */
          fdo = open(outfile, O_WRONLY | O_CREAT | O_TRUNC);
        }
        i++;
      }
    }

    dup2(fdi, STDIN_FILENO);  /* redirect stdin to file or restore stdin */
    dup2(fdo, STDOUT_FILENO); /* redirect stdout to file or restore stdout */

    pid = safe_fork(); /* create a child */

    if (pid == 0) {            /* child */
      execvp(words[0], words); /* run the command */

    } else if (pid > 0) { /* parent */
      wait(&status);
      if (WIFEXITED(status)) {
        free_wrd(words); /* free array returned by split */
        count++;         /* increment count if child exits properly */
      }
    }
    curr = curr->next;
  }
  return count;
}

/* Helper function to avoid code duplication, It clears all memory like,
 * clear_commands but it takes in a pointer to a single list. Because the two
 * lists might not have same number of elements.
 */

static void clear(Node *head) {
  Node *curr, *pre;
  if (head != NULL) {
    curr = head;
    while (curr->next != NULL) {
      pre = curr;
      curr = curr->next;
      free(pre->data);
      free(pre);
    }
    free(curr->data);
    free(curr);
  }
}

/* Helper function to clear the dynamically allocated array
 * returned by split to avoid memory leaks.
 */

void free_wrd(char *name_list[]) {
  if (name_list != NULL) {
    int i;
    i = 0;
    while (name_list[i] != NULL) {
      free(name_list[i]);
      i++;
    }
    free(name_list[i]);
    free(name_list);
  }
}

/* Helper function to to remove </> and input/output filenames
 * from the dynamically allocated array returned by split to
 * avoid memory leaks.
 */

char **remove_word(int n, char **array) {
  char *temp, *temp2;
  int i;
  i = 0;
  temp = array[n];
  temp2 = array[n + 1];
  while (array[i + 1] != NULL) {
    if (i >= n) {
      array[i] = array[i + 2];
    }
    i++;
  }
  free(temp);
  free(temp2);
  return array;
}

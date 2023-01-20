#include <stdio.h>
#include <assert.h>
#include "simple-sserver.h"

/* CMSC 216, Spring 2021, Project #8
 * Public test 10 (public10.c)
 *
 * Tests calling read_commands() with an invalid (missing) file as one of
 * the filename arguments.  (Note that the program must exit with the
 * correct exit status for the test to pass.)
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
  Commands commands;

  printf("No assertion remains unsatisfied!\n");

  commands= read_commands("a-file-that-does-not-exist", "public10-test-cmds");

  /* this should never execute, but it's needed to make the compiler happy,
   * otherwise commands is a variable that is set but never used
   */
  compile_program(commands);

  /* if we get here and print this the test should FAIL, because
   * read_commands() SHOULD have caused the entire program to exit without
   * doing anything when trying to open the missing file
   */
  printf("You lose!!!\n");
  
  return 0;
}

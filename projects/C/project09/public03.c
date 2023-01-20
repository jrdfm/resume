#include <stdio.h>
#include <assert.h>
#include "simple-sserver.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #8
 * Public test 3 (public03.c)
 *
 * Tests executing multiple compilation commands and verifies that there are
 * no errors in the heap or memory leaks.  (In this test the compilation
 * commands do not run the compiler- they're just simple echo commands that
 * print something, which just tests the ability to execute multiple
 * compilation commands.)  (This is the same as the previous test except for
 * freeing the data read, and checking the consistency and correctness of
 * the heap afterwards.)
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the Project #7 assignment for explanation).  Then
 * REMOVE this option when you are done using valgrind, or you will NOT
 * actually be using the memory checking functions at all.  (This option
 * DISABLES them.)
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Commands commands;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  commands= read_commands("public03.compile-cmds", "public03.test-cmds");

  assert(compile_program(commands) == 1);

  /* now free the commands */
  clear_commands(&commands);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("No assertion remains unsatisfied!\n");
  
  return 0;
}
